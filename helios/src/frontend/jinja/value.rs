/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

use std::cell::{Cell, RefCell};
use std::fmt::Write;
use std::rc::Rc;

use serde_json::Value as Json;

use super::{Error, Result};

#[derive(Clone, Debug, Default)]
pub(super) enum Value<'a> {
    #[default]
    Undefined,
    None,
    Bool(bool),
    Int(i64),
    UInt(u64),
    Float(f64),
    Str(&'a str),
    String(Rc<(String, Vec<usize>, Vec<(usize, bool)>)>),
    Array(&'a [Json]),
    Object(&'a serde_json::Map<String, Json>),
    List(Rc<Vec<Value<'a>>>),
    Map(Rc<Vec<(String, Value<'a>)>>),
    Namespace(Rc<RefCell<Vec<(String, Value<'a>)>>>),
    Loop(Rc<LoopValue<'a>>),
    Macro(u32),
    Builtin,
}

#[derive(Debug)]
pub(super) struct LoopValue<'a> {
    pub items: Vec<Value<'a>>,
    pub index: Cell<usize>,
    pub trust: bool,
}

impl<'a> Value<'a> {
    pub fn json(value: &'a Json) -> Self {
        match value {
            Json::Null => Self::None,
            Json::Bool(v) => Self::Bool(*v),
            Json::Number(v) => v
                .as_i64()
                .map(Self::Int)
                .or_else(|| v.as_u64().map(Self::UInt))
                .unwrap_or_else(|| Self::Float(v.as_f64().unwrap())),
            Json::String(v) => Self::Str(v),
            Json::Array(v) => Self::Array(v),
            Json::Object(v) => Self::Object(v),
        }
    }

    pub fn owned(value: Json) -> Self {
        match value {
            Json::Null => Self::None,
            Json::Bool(v) => Self::Bool(v),
            Json::Number(v) => v
                .as_i64()
                .map(Self::Int)
                .or_else(|| v.as_u64().map(Self::UInt))
                .unwrap_or_else(|| Self::Float(v.as_f64().unwrap())),
            Json::String(v) => Self::string(v),
            Json::Array(v) => Self::list(v.into_iter().map(Self::owned).collect()),
            Json::Object(v) => Self::Map(Rc::new(
                v.into_iter().map(|(k, v)| (k, Self::owned(v))).collect(),
            )),
        }
    }

    pub fn string(value: String) -> Self {
        Self::marked(value, Vec::new())
    }
    pub fn marked(value: String, hints: Vec<usize>) -> Self {
        Self::String(Rc::new((value, hints, Vec::new())))
    }
    pub fn marked_runs(value: String, hints: Vec<usize>, runs: Vec<(usize, bool)>) -> Self {
        Self::String(Rc::new((value, hints, runs)))
    }
    pub fn carries_untrusted(&self) -> bool {
        match self {
            Self::String(s) => s.2.iter().any(|(_, trusted)| !trusted),
            Self::List(items) => items.iter().any(Self::carries_untrusted),
            Self::Map(pairs) => pairs.iter().any(|(_, value)| value.carries_untrusted()),
            Self::Namespace(pairs) => pairs
                .borrow()
                .iter()
                .any(|(_, value)| value.carries_untrusted()),
            _ => false,
        }
    }
    pub fn baked_untrusted(value: Self) -> Self {
        match value {
            Self::Str(s) => {
                let runs = vec![(s.len(), false)];
                Self::marked_runs(s.to_owned(), Vec::new(), runs)
            }
            Self::String(s) => {
                if s.2.is_empty() {
                    Self::String(Rc::new((
                        s.0.clone(),
                        s.1.clone(),
                        vec![(s.0.len(), false)],
                    )))
                } else {
                    Self::String(s)
                }
            }
            Self::Array(items) => Self::list(
                items
                    .iter()
                    .map(|item| Self::baked_untrusted(Self::json(item)))
                    .collect(),
            ),
            Self::Object(members) => Self::Map(Rc::new(
                members
                    .iter()
                    .map(|(key, item)| (key.clone(), Self::baked_untrusted(Self::json(item))))
                    .collect(),
            )),
            Self::List(items) => {
                Self::list(items.iter().cloned().map(Self::baked_untrusted).collect())
            }
            Self::Map(pairs) => Self::Map(Rc::new(
                pairs
                    .iter()
                    .map(|(key, item)| (key.clone(), Self::baked_untrusted(item.clone())))
                    .collect(),
            )),
            Self::Namespace(pairs) => Self::Namespace(Rc::new(RefCell::new(
                pairs
                    .borrow()
                    .iter()
                    .map(|(key, item)| (key.clone(), Self::baked_untrusted(item.clone())))
                    .collect(),
            ))),
            other => other,
        }
    }
    pub fn hints(&self) -> &[usize] {
        match self {
            Self::String(s) => &s.1,
            _ => &[],
        }
    }
    pub fn append(&self, output: &mut String, hints: &mut Vec<usize>) {
        let start = output.len();
        hints.extend(self.hints().iter().map(|end| start + end));
        self.write(output);
    }
    pub fn append_annotated(
        &self,
        output: &mut String,
        hints: &mut Vec<usize>,
        runs: &mut Vec<(usize, bool)>,
        trust: bool,
    ) {
        let start = output.len();
        hints.extend(self.hints().iter().map(|end| start + end));
        if let Self::String(s) = self {
            if !s.2.is_empty() {
                output.push_str(&s.0);
                for (end, trusted) in &s.2 {
                    let end = start + end;
                    match runs.last_mut() {
                        Some(last) if last.1 == *trusted => last.0 = end,
                        _ => runs.push((end, *trusted)),
                    }
                }
                return;
            }
        }
        self.write(output);
        let end = output.len();
        match runs.last_mut() {
            Some(last) if last.1 == trust => last.0 = end,
            _ => runs.push((end, trust)),
        }
    }
    pub fn sliced_annotations(&self, start: usize, end: usize) -> (Vec<usize>, Vec<(usize, bool)>) {
        let mut hints = Vec::new();
        let mut runs: Vec<(usize, bool)> = Vec::new();
        if let Self::String(s) = self {
            for hint in &s.1 {
                if *hint > start {
                    hints.push((*hint - start).min(end - start));
                }
            }
            let mut cursor = 0;
            for (run_end, trusted) in &s.2 {
                let run_start = cursor;
                cursor = *run_end;
                let low = run_start.max(start);
                let high = (*run_end).min(end);
                if high > low {
                    match runs.last_mut() {
                        Some(last) if last.1 == *trusted => last.0 = high - start,
                        _ => runs.push((high - start, *trusted)),
                    }
                }
            }
        }
        (hints, runs)
    }
    pub(super) fn trust_at_byte(&self, byte: usize) -> Option<bool> {
        let Self::String(s) = self else {
            return None;
        };
        s.2.iter()
            .find(|(end, _)| byte < *end)
            .map(|(_, trusted)| *trusted)
    }
    fn annotated_char(&self, text: &str, byte: usize) -> Self {
        let character = text[byte..].chars().next().unwrap();
        match self.trust_at_byte(byte) {
            Some(trusted) => Self::marked_runs(
                character.to_string(),
                Vec::new(),
                vec![(character.len_utf8(), trusted)],
            ),
            None => Self::string(character.to_string()),
        }
    }
    pub fn list(value: Vec<Self>) -> Self {
        Self::List(Rc::new(value))
    }
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::Str(s) => Some(s),
            Self::String(s) => Some(&s.0),
            _ => None,
        }
    }
    pub fn integer(&self) -> Option<i64> {
        match self {
            Self::Int(v) => Some(*v),
            Self::UInt(v) => i64::try_from(*v).ok(),
            Self::Bool(v) => Some(*v as i64),
            _ => None,
        }
    }
    pub fn number(&self) -> Option<f64> {
        match self {
            Self::Float(v) => Some(*v),
            Self::UInt(v) => Some(*v as f64),
            _ => self.integer().map(|v| v as f64),
        }
    }
    pub fn len(&self) -> usize {
        match self {
            Self::Str(s) => s.chars().count(),
            Self::String(s) => s.0.chars().count(),
            Self::Array(v) => v.len(),
            Self::Object(v) => v.len(),
            Self::List(v) => v.len(),
            Self::Map(v) => v.len(),
            Self::Namespace(v) => v.borrow().len(),
            _ => 0,
        }
    }
    pub fn truthy(&self) -> bool {
        match self {
            Self::Undefined | Self::None => false,
            Self::Bool(v) => *v,
            Self::Int(v) => *v != 0,
            Self::UInt(v) => *v != 0,
            Self::Float(v) => *v != 0.0,
            Self::Loop(_) | Self::Macro(_) | Self::Builtin | Self::Namespace(_) => true,
            _ => self.len() != 0,
        }
    }
    pub fn mapping(&self) -> bool {
        matches!(self, Self::Object(_) | Self::Map(_))
    }
    pub fn sequence(&self) -> bool {
        matches!(self, Self::Array(_) | Self::List(_)) || self.as_str().is_some()
    }
    pub fn iterable(&self) -> bool {
        self.mapping() || self.sequence() || matches!(self, Self::Undefined)
    }

    pub fn attr(&self, name: &str) -> Self {
        match self {
            Self::Object(v) => v.get(name).map(Self::json).unwrap_or_default(),
            Self::Map(v) => v
                .iter()
                .find(|(k, _)| k == name)
                .map(|(_, v)| v.clone())
                .unwrap_or_default(),
            Self::Namespace(v) => v
                .borrow()
                .iter()
                .find(|(k, _)| k == name)
                .map(|(_, v)| v.clone())
                .unwrap_or_default(),
            Self::Loop(v) => {
                let i = v.index.get();
                match name {
                    "index0" => Self::Int(i as i64),
                    "index" => Self::Int(i as i64 + 1),
                    "length" => Self::Int(v.items.len() as i64),
                    "revindex0" => Self::Int((v.items.len() - i - 1) as i64),
                    "revindex" => Self::Int((v.items.len() - i) as i64),
                    "first" => Self::Bool(i == 0),
                    "last" => Self::Bool(i + 1 == v.items.len()),
                    "previtem" => i
                        .checked_sub(1)
                        .and_then(|i| v.items.get(i))
                        .cloned()
                        .unwrap_or_default(),
                    "nextitem" => v.items.get(i + 1).cloned().unwrap_or_default(),
                    _ => Self::Undefined,
                }
            }
            _ => Self::Undefined,
        }
    }

    pub fn index(&self, key: &Self) -> Self {
        if let Some(name) = key.as_str() {
            return self.attr(name);
        }
        let Some(mut index) = key.integer() else {
            return Self::Undefined;
        };
        if index < 0 {
            index += self.len() as i64;
        }
        let Ok(index) = usize::try_from(index) else {
            return Self::Undefined;
        };
        match self {
            Self::Array(v) => v.get(index).map(Self::json).unwrap_or_default(),
            Self::List(v) => v.get(index).cloned().unwrap_or_default(),
            _ => self
                .as_str()
                .and_then(|s| {
                    s.char_indices()
                        .nth(index)
                        .map(|(byte, _)| self.annotated_char(s, byte))
                })
                .unwrap_or_default(),
        }
    }

    pub fn iter(&self) -> Result<Vec<Self>> {
        match self {
            Self::Array(v) => Ok(v.iter().map(Self::json).collect()),
            Self::List(v) => Ok(v.as_ref().clone()),
            Self::Object(v) => Ok(v.keys().map(|s| Self::Str(s)).collect()),
            Self::Map(v) => Ok(v.iter().map(|(s, _)| Self::string(s.clone())).collect()),
            Self::Undefined | Self::None => Ok(Vec::new()),
            _ if self.as_str().is_some() => Ok(self
                .as_str()
                .unwrap()
                .char_indices()
                .map(|(byte, _)| self.annotated_char(self.as_str().unwrap(), byte))
                .collect()),
            _ => Err(Error::new("value is not iterable")),
        }
    }

    pub fn pairs(&self) -> Result<Vec<(String, Self)>> {
        match self {
            Self::Object(v) => Ok(v.iter().map(|(k, v)| (k.clone(), Self::json(v))).collect()),
            Self::Map(v) => Ok(v.as_ref().clone()),
            Self::Namespace(v) => Ok(v.borrow().clone()),
            Self::Undefined => Ok(Vec::new()),
            _ => Err(Error::new("value is not a map")),
        }
    }

    pub fn write(&self, output: &mut String) {
        match self {
            Self::Undefined => {}
            Self::None => output.push_str("None"),
            Self::Bool(v) => output.push_str(if *v { "True" } else { "False" }),
            Self::Int(v) => {
                let _ = write!(output, "{v}");
            }
            Self::UInt(v) => {
                let _ = write!(output, "{v}");
            }
            Self::Float(v) if v.is_nan() => output.push_str("nan"),
            Self::Float(v) if v.is_infinite() => {
                output.push_str(if v.is_sign_negative() { "-inf" } else { "inf" })
            }
            Self::Float(v) => output.push_str(&super::super::json::python_float(*v)),
            Self::Str(s) => output.push_str(s),
            Self::String(s) => output.push_str(&s.0),
            _ => self.write_repr(output),
        }
    }

    fn write_repr(&self, output: &mut String) {
        if let Some(s) = self.as_str() {
            output.push_str(&serde_json::to_string(s).unwrap());
        } else if self.mapping() || matches!(self, Self::Namespace(_)) {
            output.push('{');
            for (i, (k, v)) in self.pairs().unwrap().into_iter().enumerate() {
                if i != 0 {
                    output.push_str(", ");
                }
                output.push_str(&serde_json::to_string(&k).unwrap());
                output.push_str(": ");
                v.write_repr(output);
            }
            output.push('}');
        } else if self.sequence() {
            output.push('[');
            for (i, v) in self.iter().unwrap().iter().enumerate() {
                if i != 0 {
                    output.push_str(", ");
                }
                v.write_repr(output);
            }
            output.push(']');
        } else if matches!(self, Self::Loop(_) | Self::Macro(_)) {
            output.push_str("<object>");
        } else {
            self.write(output);
        }
    }

    pub fn text(&self) -> String {
        let mut s = String::new();
        self.write(&mut s);
        s
    }

    pub fn to_json(&self) -> Result<Json> {
        Ok(match self {
            Self::None | Self::Undefined => Json::Null,
            Self::Bool(v) => (*v).into(),
            Self::Int(v) => (*v).into(),
            Self::Float(v) => (*v).into(),
            Self::UInt(v) => (*v).into(),
            Self::Str(s) => (*s).into(),
            Self::String(s) => s.0.as_str().into(),
            Self::Array(v) => Json::Array(v.to_vec()),
            Self::Object(v) => Json::Object((*v).clone()),
            Self::List(v) => Json::Array(v.iter().map(Self::to_json).collect::<Result<_>>()?),
            Self::Map(_) | Self::Namespace(_) => Json::Object(
                self.pairs()?
                    .into_iter()
                    .map(|(k, v)| Ok((k, v.to_json()?)))
                    .collect::<Result<_>>()?,
            ),
            _ => return Err(Error::new("cannot serialize template object")),
        })
    }

    pub fn json_hints<'j>(&self, json: &'j Json, hints: &mut Vec<(&'j str, Vec<usize>)>) {
        match (self, json) {
            (Self::String(value), Json::String(text)) if !value.1.is_empty() => {
                hints.push((text, value.1.clone()));
            }
            (Self::List(values), Json::Array(items)) => {
                for (value, item) in values.iter().zip(items) {
                    value.json_hints(item, hints);
                }
            }
            (Self::Map(values), Json::Object(items)) => {
                for (key, item) in items {
                    if let Some((_, value)) = values.iter().rev().find(|(name, _)| name == key) {
                        value.json_hints(item, hints);
                    }
                }
            }
            (Self::Namespace(values), Json::Object(items)) => {
                for (key, value) in values.borrow().iter() {
                    value.json_hints(&items[key], hints);
                }
            }
            _ => {}
        }
    }

    pub fn equals(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Int(a), Self::Int(b)) => return a == b,
            (Self::UInt(a), Self::UInt(b)) => return a == b,
            (Self::Int(a), Self::UInt(b)) | (Self::UInt(b), Self::Int(a)) => {
                return u64::try_from(*a).ok() == Some(*b);
            }
            _ => {}
        }
        if let (Some(a), Some(b)) = (self.as_str(), other.as_str()) {
            return a == b;
        }
        if let (Some(a), Some(b)) = (self.number(), other.number()) {
            return a == b;
        }
        if matches!(
            (self, other),
            (Self::None, Self::None) | (Self::Undefined, Self::Undefined)
        ) {
            return true;
        }
        if (self.sequence() && other.sequence()) || (self.mapping() && other.mapping()) {
            return self.to_json().ok() == other.to_json().ok();
        }
        false
    }
}
