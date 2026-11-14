/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

use std::cell::{Cell, RefCell};
use std::rc::Rc;

use serde_json::{Map, Value as Json};

use super::compiler::{Binary, Instruction as I, Unary};
use super::value::{LoopValue, Value};
use super::{CompiledTemplate, Error, Mark, RenderedTemplate, Result};

struct Scope {
    undo: usize,
    id: usize,
    closures: usize,
}
struct Closure<'a> {
    definition: u32,
    scope: usize,
    captured: Option<(Vec<Value<'a>>, Vec<bool>)>,
}
struct Vm<'a> {
    template: &'a CompiledTemplate,
    slots: Vec<Value<'a>>,
    stamps: Vec<usize>,
    scope: usize,
    serial: usize,
    scopes: Vec<Scope>,
    closures: Vec<Closure<'a>>,
    undo: Vec<(u32, Value<'a>, usize, bool)>,
    stack: Vec<Value<'a>>,
    stack_trust: Vec<bool>,
    slots_trust: Vec<bool>,
    loops: Vec<Rc<LoopValue<'a>>>,
    output: String,
    output_runs: Vec<(usize, bool)>,
    captures: Vec<(String, Vec<usize>, Vec<(usize, bool)>)>,
    hint_positions: Vec<usize>,
    marks: Vec<Mark>,
    depth: usize,
    hint_texts: Vec<&'a str>,
}

pub(super) fn render(
    template: &CompiledTemplate,
    context: &Map<String, Json>,
    hints: &[(usize, usize)],
    trusted: &std::collections::HashSet<String>,
) -> Result<RenderedTemplate> {
    let mut hint_texts = Vec::new();
    for &(message, count) in hints {
        let Some(message) = context
            .get("messages")
            .and_then(|messages| messages.get(message))
        else {
            continue;
        };
        let Some(content) = message.get("content") else {
            continue;
        };
        let text = match content {
            Json::String(text) if count == 1 => Some(text.as_str()),
            Json::Array(parts) => count
                .checked_sub(1)
                .and_then(|index| parts.get(index))
                .and_then(|part| part.get("text"))
                .and_then(Json::as_str),
            _ => None,
        };
        if let Some(text) = text {
            hint_texts.push(text);
        }
    }
    let mut vm = Vm {
        template,
        slots: template
            .names
            .iter()
            .map(|s| context.get(s).map(Value::json).unwrap_or_else(|| if s == "strftime_now" { Value::Builtin } else { Value::Undefined }))
            .collect(),
        stamps: vec![0; template.names.len()],
        scope: 0,
        serial: 0,
        scopes: Vec::new(),
        closures: Vec::new(),
        undo: Vec::new(),
        stack: Vec::with_capacity(16),
        stack_trust: Vec::with_capacity(16),
        slots_trust: template.names.iter().map(|s| trusted.contains(s)).collect(),
        loops: Vec::new(),
        output: String::new(),
        output_runs: Vec::new(),
        captures: Vec::new(),
        hint_positions: Vec::new(),
        marks: Vec::new(),
        depth: 0,
        hint_texts,
    };
    vm.run(0)?;
    vm.marks.extend(vm.hint_positions.into_iter().map(Mark::Hint));
    Ok(RenderedTemplate {
        text: vm.output,
        marks: vm.marks,
        runs: vm.output_runs,
    })
}

impl<'a> Vm<'a> {
    fn pop(&mut self) -> Value<'a> {
        self.stack
            .pop()
            .expect("compiler balances the operand stack")
    }
    fn pop_trust(&mut self) -> bool {
        self.stack_trust
            .pop()
            .expect("compiler balances the operand stack")
    }
    fn pop_trusted(&mut self) -> (Value<'a>, bool) {
        let trust = self.pop_trust();
        (self.pop(), trust)
    }
    fn force_trust(value: Value<'a>, trusted: bool) -> Value<'a> {
        if trusted {
            value
        } else {
            Value::baked_untrusted(value)
        }
    }
    fn enter(&mut self) {
        self.scopes.push(Scope {
            undo: self.undo.len(),
            id: self.scope,
            closures: self.closures.len(),
        });
        self.serial += 1;
        self.scope = self.serial;
    }
    fn leave(&mut self) {
        for i in self.scopes.last().unwrap().closures..self.closures.len() {
            let closure = &self.closures[i];
            if closure.scope == self.scope && closure.captured.is_none() {
                let values = self.template.macros[closure.definition as usize]
                    .captures
                    .iter()
                    .map(|&s| (self.lexical(s, self.scope), self.lexical_trust(s, self.scope)))
                    .collect::<Vec<_>>();
                let (values, trusts): (Vec<_>, Vec<_>) = values
                    .into_iter()
                    .map(|(value, trust)| (Value::baked_untrusted(value), trust))
                    .unzip();
                self.closures[i].captured = Some((values, trusts));
            }
        }
        let scope = self.scopes.pop().unwrap();
        for (slot, value, stamp, trust) in self.undo.drain(scope.undo..).rev() {
            self.slots[slot as usize] = value;
            self.stamps[slot as usize] = stamp;
            self.slots_trust[slot as usize] = trust;
        }
        self.scope = scope.id;
    }
    fn lexical(&self, slot: u32, scope: usize) -> Value<'a> {
        let mut value = &self.slots[slot as usize];
        let mut stamp = self.stamps[slot as usize];
        if stamp > scope {
            for (s, previous, previous_stamp, _) in self.undo.iter().rev() {
                if *s == slot {
                    value = previous;
                    stamp = *previous_stamp;
                    if stamp <= scope {
                        break;
                    }
                }
            }
        }
        value.clone()
    }
    fn lexical_trust(&self, slot: u32, scope: usize) -> bool {
        let mut trust = self.slots_trust[slot as usize];
        let mut stamp = self.stamps[slot as usize];
        if stamp > scope {
            for (s, _, previous_stamp, previous_trust) in self.undo.iter().rev() {
                if *s == slot {
                    trust = *previous_trust;
                    stamp = *previous_stamp;
                    if stamp <= scope {
                        break;
                    }
                }
            }
        }
        trust
    }
    fn store(&mut self, slot: u32, value: Value<'a>, trust: bool) {
        let value = self.hinted(value);
        let index = slot as usize;
        if self.scope != 0 && self.stamps[index] != self.scope {
            self.undo.push((
                slot,
                std::mem::take(&mut self.slots[index]),
                self.stamps[index],
                self.slots_trust[index],
            ));
            self.stamps[index] = self.scope;
        }
        self.slots[index] = value;
        self.slots_trust[index] = trust;
    }
    fn iteration(&mut self, id: u32) -> Result<()> {
        let value = self.loops.last().unwrap().clone();
        let item = value.items[value.index.get()].clone();
        let trust = value.trust;
        self.store(self.template.loop_symbol, Value::Loop(value), trust);
        self.loop_variables(id, item, trust)
    }
    fn loop_variables(&mut self, id: u32, item: Value<'a>, trust: bool) -> Result<()> {
        let vars = &self.template.loops[id as usize].variables;
        if vars.len() == 1 {
            self.store(vars[0], item, trust);
        } else {
            let items = item.iter()?;
            if items.len() != vars.len() {
                return Err(Error::new("cannot unpack sequence of different length"));
            }
            for (&slot, value) in vars.iter().zip(items) {
                self.store(slot, value, trust);
            }
        }
        Ok(())
    }
    fn hinted(&self, value: Value<'a>) -> Value<'a> {
        if let Value::Str(text) = value
            && self
                .hint_texts
                .iter()
                .any(|hint| !text.is_empty() && std::ptr::eq(*hint, text))
        {
            return Value::marked(text.to_owned(), vec![text.len()]);
        }
        value
    }
    fn emit(&mut self, value: Value<'a>, trust: bool) {
        self.hinted(value).append_annotated(
            &mut self.output,
            &mut self.hint_positions,
            &mut self.output_runs,
            trust,
        );
    }
    fn capture(&mut self) {
        self.captures.push((
            std::mem::take(&mut self.output),
            std::mem::take(&mut self.hint_positions),
            std::mem::take(&mut self.output_runs),
        ));
    }
    fn end_capture(&mut self) -> Value<'a> {
        let (output, hints, runs) = self.captures.pop().unwrap();
        let text = std::mem::replace(&mut self.output, output);
        let positions = std::mem::replace(&mut self.hint_positions, hints);
        let runs = std::mem::replace(&mut self.output_runs, runs);
        Value::marked_runs(text, positions, runs)
    }
    fn run(&mut self, start: u32) -> Result<()> {
        self.depth += 1;
        if self.depth > 128 {
            return Err(Error::new("macro recursion limit exceeded"));
        }
        let mut pc = start as usize;
        let result = (|| {
            while pc < self.template.code.len() {
                let op = self.template.code[pc];
                pc += 1;
                match op {
                    I::Text(id) => {
                        self.output
                            .push_str(self.template.constants[id as usize].as_str().unwrap());
                        self.output_runs.push((self.output.len(), true));
                    }
                    I::Constant(id) => {
                        self.stack
                            .push(Value::json(&self.template.constants[id as usize]));
                        self.stack_trust.push(true);
                    }
                    I::Load(id) => {
                        self.stack.push(self.slots[id as usize].clone());
                        self.stack_trust.push(self.slots_trust[id as usize]);
                    }
                    I::Store(id) => {
                        let trust = self.pop_trust();
                        let v = self.pop();
                        self.store(id, v, trust);
                    }
                    I::StoreAttr(slot, name) | I::AppendAttr(slot, name) => {
                        let trust = self.pop_trust();
                        let value = self.pop();
                        let value = if trust {
                            value
                        } else {
                            Value::baked_untrusted(value)
                        };
                        let Value::Namespace(ns) = &self.slots[slot as usize] else {
                            return Err(Error::new(
                                "cannot assign attribute of non-namespace object",
                            ));
                        };
                        let mut ns = ns.borrow_mut();
                        let name = &self.template.names[name as usize];
                        if let Some((_, old)) = ns.iter_mut().find(|(key, _)| key == name) {
                            if matches!(op, I::AppendAttr(..)) {
                                match (old, value) {
                                    (Value::List(items), value)
                                        if value.sequence() && value.as_str().is_none() =>
                                    {
                                        Rc::make_mut(items).extend(value.iter()?)
                                    }
                                    (old, value)
                                        if old.as_str().is_some() && value.as_str().is_some() =>
                                    {
                                        let mut text = String::new();
                                        let mut hints = Vec::new();
                                        let mut runs = Vec::new();
                                        old.append_annotated(
                                            &mut text, &mut hints, &mut runs, true,
                                        );
                                        value.append_annotated(
                                            &mut text, &mut hints, &mut runs, true,
                                        );
                                        *old = Value::marked_runs(text, hints, runs);
                                    }
                                    (old, value) => {
                                        *old = binary(Binary::Add, std::mem::take(old), value)?
                                    }
                                }
                            } else {
                                *old = value;
                            }
                        } else if matches!(op, I::AppendAttr(..)) {
                            return Err(Error::new("undefined namespace attribute"));
                        } else {
                            ns.push((name.clone(), value));
                        }
                    }
                    I::Attr(name) => {
                        let trust = self.pop_trust();
                        let v = self.pop();
                        if matches!(v, Value::Undefined) {
                            return Err(Error::new("undefined value"));
                        }
                        let value = self.hinted(v.attr(&self.template.names[name as usize]));
                        self.stack.push(value);
                        self.stack_trust.push(trust);
                    }
                    I::Index => {
                        let _ = self.pop_trust();
                        let key = self.pop();
                        let trust = self.pop_trust();
                        let v = self.pop();
                        if matches!(v, Value::Undefined) {
                            return Err(Error::new("undefined value"));
                        }
                        let value = self.hinted(v.index(&key));
                        self.stack.push(value);
                        self.stack_trust.push(trust);
                    }
                    I::Slice => {
                        let _ = self.pop_trust();
                        let step = self.pop();
                        let _ = self.pop_trust();
                        let end = self.pop();
                        let _ = self.pop_trust();
                        let begin = self.pop();
                        let trust = self.pop_trust();
                        let v = self.pop();
                        let sliced = slice(v, begin, end, step)?;
                        self.stack.push(Self::force_trust(sliced, trust));
                        self.stack_trust.push(trust);
                    }
                    I::List(n) => {
                        let trusts = self.stack_trust.split_off(self.stack_trust.len() - n as usize);
                        let items = self.stack.split_off(self.stack.len() - n as usize);
                        let trusted = trusts.iter().all(|trusted| *trusted);
                        let items = items
                            .into_iter()
                            .zip(trusts)
                            .map(|(value, trust)| if trust { value } else { Value::baked_untrusted(value) })
                            .collect();
                        self.stack.push(Value::list(items));
                        self.stack_trust.push(trusted);
                    }
                    I::Map(n) => {
                        let trusts = self.stack_trust.split_off(self.stack_trust.len() - 2 * n as usize);
                        let values = self.stack.split_off(self.stack.len() - 2 * n as usize);
                        let mut pairs = Vec::with_capacity(n as usize);
                        let mut trusted = true;
                        let mut entries = values.into_iter().zip(trusts);
                        while let Some((key, key_trust)) = entries.next() {
                            let (value, value_trust) = entries.next().unwrap();
                            trusted &= key_trust & value_trust;
                            let value = if value_trust {
                                value
                            } else {
                                Value::baked_untrusted(value)
                            };
                            pairs.push((key.text(), value));
                        }
                        self.stack.push(Value::Map(Rc::new(pairs)));
                        self.stack_trust.push(trusted);
                    }
                    I::Unary(op) => {
                        let trust = self.pop_trust();
                        let v = self.pop();
                        let (value, trusted) = match op {
                            Unary::Not => (Value::Bool(!v.truthy()), true),
                            Unary::Pos => (v, trust),
                            Unary::Neg => match v {
                                Value::Int(v) => (Value::Int(
                                    v.checked_neg()
                                        .ok_or_else(|| Error::new("integer overflow"))?,
                                ), true),
                                _ => (Value::Float(
                                    -v.number()
                                        .ok_or_else(|| Error::new("cannot negate non-number"))?,
                                ), true),
                            },
                        };
                        self.stack.push(value);
                        self.stack_trust.push(trusted);
                    }
                    I::Binary(op) => {
                        let b_trust = self.pop_trust();
                        let b = self.pop();
                        let a_trust = self.pop_trust();
                        let a = self.pop();
                        let mut annotated = false;
                        let result = if matches!(op, Binary::Concat | Binary::Add)
                            && a.as_str().is_some()
                            && b.as_str().is_some()
                        {
                            annotated = true;
                            let mut text = String::new();
                            let mut hints = Vec::new();
                            let mut runs = Vec::new();
                            a.append_annotated(&mut text, &mut hints, &mut runs, a_trust);
                            b.append_annotated(&mut text, &mut hints, &mut runs, b_trust);
                            Value::marked_runs(text, hints, runs)
                        } else {
                            binary(op, a, b)?
                        };
                        let trusted = match op {
                            Binary::Eq
                            | Binary::Ne
                            | Binary::In
                            | Binary::NotIn
                            | Binary::Lt
                            | Binary::Le
                            | Binary::Gt
                            | Binary::Ge => true,
                            Binary::Concat | Binary::Add if annotated => {
                                !result.carries_untrusted()
                            }
                            _ => a_trust && b_trust,
                        };
                        let result = Self::force_trust(result, trusted);
                        self.stack.push(result);
                        self.stack_trust.push(trusted);
                    }
                    I::Test(id) | I::Filter(id) | I::Method(id) | I::Call(id) => {
                        let call = &self.template.calls[id as usize];
                        let count = call.positional + call.keywords.len();
                        let arg_trusts = self.stack_trust.split_off(self.stack_trust.len() - count);
                        let arguments = self.stack.split_off(self.stack.len() - count);
                        let name = self.template.names[call.name as usize].as_str();
                        let (args, kw) = arguments.split_at(call.positional);
                        let (arg_trusts, kw_trusts) = arg_trusts.split_at(call.positional);
                        let kwargs: Vec<_> = call
                            .keywords
                            .iter()
                            .zip(kw)
                            .zip(kw_trusts)
                            .map(|((key, v), trust)| {
                                (self.template.names[*key as usize].as_str(), v.clone(), *trust)
                            })
                            .collect();
                        let kw_named: Vec<(&str, Value<'a>)> = kwargs
                            .iter()
                            .map(|(name, value, _)| (*name, value.clone()))
                            .collect();
                        let arguments_trusted = arg_trusts.iter().chain(kw_trusts.iter()).all(|trusted| *trusted);
                        let mut input_trust = true;
                        let mut input_untrusted = false;
                        let mut macro_result = false;
                        let result = match op {
                            I::Call(_) => {
                                if let Value::Macro(id) = self.slots[call.name as usize] {
                                    let direct =
                                        matches!(self.template.code.get(pc), Some(I::Emit));
                                    let value = self.macro_call(
                                        id,
                                        args,
                                        &kwargs,
                                        arg_trusts,
                                        direct,
                                    )?;
                                    if direct {
                                        pc += 1;
                                        continue;
                                    }
                                    macro_result = true;
                                    value
                                } else {
                                    function(name, args, &kw_named)?
                                }
                            }
                            I::Test(_) => {
                                let (v, _) = self.pop_trusted();
                                Value::Bool(test(name, &v, args)?)
                            }
                            I::Filter(_) => {
                                let (v, trust) = self.pop_trusted();
                                input_trust = trust;
                                input_untrusted = v.carries_untrusted();
                                filter(name, v, args, &kw_named, !self.hint_texts.is_empty())?
                            }
                            _ => {
                                let (v, trust) = self.pop_trusted();
                                input_trust = trust;
                                input_untrusted = v.carries_untrusted();
                                method(name, v, args, &kw_named)?
                            }
                        };
                        let trusted = match op {
                            I::Test(_) => true,
                            I::Call(_) if macro_result => !result.carries_untrusted(),
                            I::Call(_) => arguments_trusted && !result.carries_untrusted(),
                            _ => {
                                arguments_trusted
                                    && input_trust
                                    && !input_untrusted
                                    && !result.carries_untrusted()
                            }
                        };
                        let result = Self::force_trust(result, trusted);
                        self.stack.push(result);
                        self.stack_trust.push(trusted);
                    }
                    I::Emit => {
                        let trust = self.pop_trust();
                        let v = self.pop();
                        self.emit(v, trust);
                    }
                    I::Join(count, strict) | I::EmitJoined(count, strict) => {
                        let start = self.stack.len() - count as usize;
                        let trusts = self.stack_trust.split_off(start);
                        let values = &self.stack[start..];
                        if strict && values.iter().any(|v| v.as_str().is_none()) {
                            return Err(Error::new("string addition requires strings"));
                        }
                        let size = values.iter().map(|v| v.as_str().map_or(0, str::len)).sum();
                        if matches!(op, I::EmitJoined(..)) {
                            self.output.reserve(size);
                            for (index, trust) in (start..self.stack.len()).zip(trusts) {
                                let value = self.stack[index].clone();
                                self.emit(value, trust);
                            }
                            self.stack.truncate(start);
                        } else {
                            let mut text = String::with_capacity(size);
                            let mut hints = Vec::new();
                            let mut runs = Vec::new();
                            for (v, trust) in self.stack.drain(start..).zip(trusts) {
                                v.append_annotated(&mut text, &mut hints, &mut runs, trust);
                            }
                            let trusted = runs.iter().all(|(_, trusted)| *trusted);
                            self.stack.push(Value::marked_runs(text, hints, runs));
                            self.stack_trust.push(trusted);
                        }
                    }
                    I::Drop => {
                        let _ = self.pop_trust();
                        self.pop();
                    }
                    I::Jump(target) => pc = target as usize,
                    I::JumpFalse(target) => {
                        let _ = self.pop_trust();
                        if !self.pop().truthy() {
                            pc = target as usize;
                        }
                    }
                    I::And(target) => {
                        if !self.stack.last().unwrap().truthy() {
                            pc = target as usize;
                        } else {
                            let _ = self.pop_trust();
                            self.pop();
                        }
                    }
                    I::Or(target) => {
                        if self.stack.last().unwrap().truthy() {
                            pc = target as usize;
                        } else {
                            let _ = self.pop_trust();
                            self.pop();
                        }
                    }
                    I::For(id) => {
                        let trust = self.pop_trust();
                        let mut items = self.pop().iter()?;
                        if let Some(filter) = self.template.loops[id as usize].filter {
                            self.enter();
                            let mut selected = Vec::new();
                            for item in items {
                                self.loop_variables(id, item.clone(), trust)?;
                                self.run(filter)?;
                                let _ = self.pop_trust();
                                if self.pop().truthy() {
                                    selected.push(item);
                                }
                            }
                            self.leave();
                            items = selected;
                        }
                        let empty = items.is_empty();
                        self.loops.push(Rc::new(LoopValue {
                            items,
                            index: Cell::new(0),
                            trust,
                        }));
                        self.enter();
                        if empty {
                            pc = self.template.loops[id as usize].empty as usize;
                        } else {
                            self.iteration(id)?;
                        }
                    }
                    I::Next(id) => {
                        let lp = self.loops.last().unwrap();
                        let next = lp.index.get() + 1;
                        lp.index.set(next);
                        if next < lp.items.len() {
                            self.leave();
                            self.enter();
                            self.iteration(id)?;
                            pc = self.template.loops[id as usize].body as usize;
                        } else {
                            pc = self.template.loops[id as usize].end as usize;
                        }
                    }
                    I::EndLoop => {
                        self.loops.pop();
                        self.leave();
                    }
                    I::Break(id) => pc = self.template.loops[id as usize].end as usize,
                    I::Continue(id) => pc = self.template.loops[id as usize].next as usize,
                    I::Macro(slot, id) => {
                        let closure = self.closures.len() as u32;
                        self.closures.push(Closure {
                            definition: id,
                            scope: self.scope,
                            captured: None,
                        });
                        self.store(slot, Value::Macro(closure), true);
                    }
                    I::Return => break,
                    I::Capture => self.capture(),
                    I::EndCapture => {
                        let value = self.end_capture();
                        let trusted = !value.carries_untrusted();
                        self.stack.push(value);
                        self.stack_trust.push(trusted);
                    }
                    I::Mark => self.marks.push(Mark::Byte(self.output.len())),
                    I::MarkPosition => {
                        let _ = self.pop_trust();
                        let Value::Int(v) = self.pop() else {
                            return Err(Error::new("mark position must be a nonnegative integer"));
                        };
                        let v = usize::try_from(v).map_err(|_| {
                            Error::new("mark position must be a nonnegative integer")
                        })?;
                        self.marks.push(Mark::Position(v));
                    }
                }
            }
            Ok(())
        })();
        self.depth -= 1;
        result.map_err(|mut error: Error| {
            if error.offset == 0 {
                error.offset = self.template.spans[pc.saturating_sub(1)];
            }
            error
        })
    }
    fn macro_call(
        &mut self,
        id: u32,
        args: &[Value<'a>],
        kwargs: &[(&str, Value<'a>, bool)],
        arg_trusts: &[bool],
        direct: bool,
    ) -> Result<Value<'a>> {
        let closure = &self.closures[id as usize];
        let scope = closure.scope;
        let mac = &self.template.macros[closure.definition as usize];
        if args.len() > mac.parameters.len() {
            return Err(Error::new("too many arguments to macro"));
        }
        self.enter();
        for (i, &slot) in mac.captures.iter().enumerate() {
            let (value, trust) = if let Some((captured, trusts)) = &self.closures[id as usize].captured
            {
                (captured[i].clone(), trusts[i])
            } else {
                let trust = self.lexical_trust(slot, scope);
                (self.lexical(slot, scope), trust)
            };
            self.store(slot, value, trust);
        }
        for (i, &slot) in mac.parameters.iter().enumerate() {
            let name = self.template.names[slot as usize].as_str();
            let value = if let Some(value) = args.get(i) {
                let trust = arg_trusts.get(i).copied().unwrap_or(true);
                (value.clone(), trust)
            } else if let Some((_, value, trust)) = kwargs.iter().find(|(k, _, _)| *k == name) {
                (value.clone(), *trust)
            } else if let Some((_, pc)) = mac.defaults.iter().find(|(s, _)| *s == slot) {
                self.run(*pc)?;
                self.pop_trusted()
            } else {
                (Value::Undefined, true)
            };
            self.store(slot, value.0, value.1);
        }
        for (name, _, _) in kwargs {
            if !mac
                .parameters
                .iter()
                .any(|&s| self.template.names[s as usize] == *name)
            {
                return Err(Error::new(format!("unknown macro argument {name}")));
            }
        }
        if !direct {
            self.capture();
        }
        self.run(mac.entry)?;
        self.leave();
        Ok(if !direct {
            self.end_capture()
        } else {
            Value::Undefined
        })
    }
}

fn binary<'a>(op: Binary, a: Value<'a>, b: Value<'a>) -> Result<Value<'a>> {
    use Binary as B;
    match op {
        B::Eq => return Ok(Value::Bool(a.equals(&b))),
        B::Ne => return Ok(Value::Bool(!a.equals(&b))),
        B::In | B::NotIn => {
            let found = if let Some(text) = b.as_str() {
                text.contains(
                    a.as_str()
                        .ok_or_else(|| Error::new("string membership requires a string"))?,
                )
            } else if b.mapping() {
                a.as_str()
                    .is_some_and(|key| !matches!(b.attr(key), Value::Undefined))
            } else {
                b.iter()?.iter().any(|v| v.equals(&a))
            };
            return Ok(Value::Bool(found != matches!(op, B::NotIn)));
        }
        B::Concat => {
            let mut text = String::new();
            let mut hints = Vec::new();
            a.append(&mut text, &mut hints);
            b.append(&mut text, &mut hints);
            return Ok(Value::marked(text, hints));
        }
        B::Add if a.as_str().is_some() && b.as_str().is_some() => {
            let mut text = String::new();
            let mut hints = Vec::new();
            a.append(&mut text, &mut hints);
            b.append(&mut text, &mut hints);
            return Ok(Value::marked(text, hints));
        }
        B::Add if a.sequence() && b.sequence() && a.as_str().is_none() && b.as_str().is_none() => {
            let mut items = a.iter()?;
            items.extend(b.iter()?);
            return Ok(Value::list(items));
        }
        B::Mul if a.as_str().is_some() && b.integer().is_some() => {
            return Ok(Value::string(
                a.as_str()
                    .unwrap()
                    .repeat(b.integer().unwrap().max(0) as usize),
            ));
        }
        B::Lt | B::Le | B::Gt | B::Ge => {
            let ordering = if let (Some(a), Some(b)) = (a.as_str(), b.as_str()) {
                Some(a.cmp(b))
            } else if let (Some(a), Some(b)) = (a.number(), b.number()) {
                a.partial_cmp(&b)
            } else {
                None
            };
            let ordering = ordering.ok_or_else(|| Error::new("cannot compare values"))?;
            return Ok(Value::Bool(match op {
                B::Lt => ordering.is_lt(),
                B::Le => ordering.is_le(),
                B::Gt => ordering.is_gt(),
                _ => ordering.is_ge(),
            }));
        }
        _ => {}
    }
    if let (Some(a), Some(b)) = (a.integer(), b.integer()) {
        let value = match op {
            B::Add => a.checked_add(b),
            B::Sub => a.checked_sub(b),
            B::Mul => a.checked_mul(b),
            B::FloorDiv if b != 0 => a.checked_div_euclid(b),
            B::Rem if b != 0 => a.checked_rem_euclid(b),
            B::Pow if b >= 0 => u32::try_from(b).ok().and_then(|b| a.checked_pow(b)),
            _ => None,
        };
        if let Some(value) = value {
            return Ok(Value::Int(value));
        }
    }
    let a = a
        .number()
        .ok_or_else(|| Error::new("arithmetic requires compatible numbers or sequences"))?;
    let b = b
        .number()
        .ok_or_else(|| Error::new("arithmetic requires compatible numbers or sequences"))?;
    if b == 0.0 && matches!(op, B::Div | B::FloorDiv | B::Rem) {
        return Err(Error::new("division by zero"));
    }
    Ok(Value::Float(match op {
        B::Add => a + b,
        B::Sub => a - b,
        B::Mul => a * b,
        B::Div => a / b,
        B::FloorDiv => (a / b).floor(),
        B::Rem => a - b * (a / b).floor(),
        B::Pow => a.powf(b),
        _ => unreachable!(),
    }))
}

fn slice<'a>(
    value: Value<'a>,
    begin: Value<'a>,
    end: Value<'a>,
    step: Value<'a>,
) -> Result<Value<'a>> {
    let n = value.len() as i64;
    let integer = |v: &Value<'_>, default| match v {
        Value::None => Ok(default),
        Value::Int(v) => Ok(*v),
        _ => Err(Error::new("slice indices must be integers")),
    };
    let step = integer(&step, 1)?;
    if step == 0 {
        return Err(Error::new("slice step cannot be zero"));
    }
    let bound = |v: &Value<'_>, default| -> Result<i64> {
        if matches!(v, Value::None) {
            return Ok(default);
        }
        let mut v = integer(v, default)?;
        if v < 0 {
            v += n;
        }
        Ok(if step < 0 {
            v.clamp(-1, n - 1)
        } else {
            v.clamp(0, n)
        })
    };
    let mut i = bound(&begin, if step < 0 { n - 1 } else { 0 })?;
    let end = bound(&end, if step < 0 { -1 } else { n })?;
    let mut indices = Vec::new();
    while if step < 0 { i > end } else { i < end } {
        indices.push(i);
        i = i
            .checked_add(step)
            .ok_or_else(|| Error::new("slice index overflow"))?;
    }
    if let Some(text) = value.as_str() {
        let mut output = String::new();
        let mut hints = Vec::new();
        let mut runs: Vec<(usize, bool)> = Vec::new();
        for index in indices {
            let Ok(index) = usize::try_from(index) else {
                continue;
            };
            let Some((byte, character)) = text.char_indices().nth(index) else {
                continue;
            };
            output.push(character);
            let output_end = output.len();
            if let Value::String(source) = &value {
                if source
                    .1
                    .iter()
                    .any(|hint| *hint == byte + character.len_utf8())
                {
                    hints.push(output_end);
                }
                if let Some(trusted) = value.trust_at_byte(byte) {
                    match runs.last_mut() {
                        Some(last) if last.1 == trusted => last.0 = output_end,
                        _ => runs.push((output_end, trusted)),
                    }
                }
            }
        }
        Ok(Value::marked_runs(output, hints, runs))
    } else {
        Ok(Value::list(
            indices
                .into_iter()
                .map(|index| value.index(&Value::Int(index)))
                .collect(),
        ))
    }
}

#[cfg(not(windows))]
use libc::strftime;
// The libc crate declares no strftime for Windows; the C runtime provides it.
#[cfg(windows)]
unsafe extern "C" {
    fn strftime(s: *mut libc::c_char, max: libc::size_t, format: *const libc::c_char, tm: *const libc::tm) -> libc::size_t;
}

fn json_object_members(source: &str) -> Result<Vec<(String, serde_json::Value, String)>> {
    use serde::de::{MapAccess, Visitor};
    use serde::Deserializer;

    struct Members;
    impl<'de> Visitor<'de> for Members {
        type Value = Vec<(String, serde_json::Value, String)>;

        fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            formatter.write_str("a JSON object")
        }

        fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> std::result::Result<Self::Value, A::Error> {
            let mut members = Vec::new();
            while let Some((key, raw)) = map.next_entry::<String, Box<serde_json::value::RawValue>>()? {
                let value: serde_json::Value = serde_json::from_str(raw.get()).map_err(serde::de::Error::custom)?;
                members.push((key, value, raw.get().to_owned()));
            }
            Ok(members)
        }
    }

    let mut decoder = serde_json::Deserializer::from_str(source);
    let members = decoder.deserialize_map(Members).map_err(|error| Error::new(format!("invalid JSON object: {error}")))?;
    decoder.end().map_err(|error| Error::new(format!("invalid JSON object: {error}")))?;
    Ok(members)
}

fn function<'a>(name: &str, args: &[Value<'a>], kw: &[(&str, Value<'a>)]) -> Result<Value<'a>> {
    match name {
        "json_object_members" => {
            if args.len() != 1 || !kw.is_empty() {
                return Err(Error::new("json_object_members expects one JSON object string"));
            }
            let source = args[0].as_str().ok_or_else(|| Error::new("json_object_members expects a JSON object string"))?;
            Ok(Value::list(json_object_members(source)?.into_iter().map(|(key, value, raw)| {
                Value::list(vec![Value::string(key), Value::owned(value), Value::string(raw)])
            }).collect()))
        }
        "strftime_now" => {
            let format = args.first().and_then(Value::as_str).ok_or_else(|| Error::new("strftime_now requires a format string"))?;
            let format = std::ffi::CString::new(format).map_err(|_| Error::new("strftime_now format contains NUL"))?;
            let mut output = vec![0u8; 4096];
            let mut local = std::mem::MaybeUninit::<libc::tm>::uninit();
            let now = unsafe { libc::time(std::ptr::null_mut()) };
            let length = unsafe {
                #[cfg(not(windows))]
                if libc::localtime_r(&now, local.as_mut_ptr()).is_null() { return Err(Error::new("local time is unavailable")); }
                #[cfg(windows)]
                if libc::localtime_s(local.as_mut_ptr(), &now) != 0 { return Err(Error::new("local time is unavailable")); }
                strftime(output.as_mut_ptr().cast(), output.len(), format.as_ptr(), local.as_ptr())
            };
            if length == 0 && !format.as_bytes().is_empty() { return Err(Error::new("strftime_now output exceeds 4095 bytes")); }
            output.truncate(length);
            Ok(Value::string(String::from_utf8(output).map_err(|_| Error::new("strftime_now output is not UTF-8"))?))
        }
        "namespace" => {
            let mut pairs = if let Some(v) = args.first() {
                v.pairs()?
            } else {
                Vec::new()
            };
            pairs.extend(kw.iter().map(|(k, v)| ((*k).into(), v.clone())));
            Ok(Value::Namespace(Rc::new(RefCell::new(pairs))))
        }
        "range" => {
            if args.is_empty() || args.len() > 3 {
                return Err(Error::new("range expects one to three arguments"));
            }
            let ints = args
                .iter()
                .map(|v| {
                    v.integer()
                        .ok_or_else(|| Error::new("range expects integers"))
                })
                .collect::<Result<Vec<_>>>()?;
            let (mut i, end, step) = match ints.as_slice() {
                [end] => (0, *end, 1),
                [start, end] => (*start, *end, 1),
                [start, end, step] => (*start, *end, *step),
                _ => unreachable!(),
            };
            if step == 0 {
                return Err(Error::new("range step cannot be zero"));
            }
            let mut items = Vec::new();
            while if step < 0 { i > end } else { i < end } {
                items.push(Value::Int(i));
                i = i
                    .checked_add(step)
                    .ok_or_else(|| Error::new("range overflow"))?;
            }
            Ok(Value::list(items))
        }
        "raise_exception" => Err(Error::new(
            args.first().map(Value::text).unwrap_or_default(),
        )),
        "dict" => Ok(Value::Map(Rc::new(
            kw.iter().map(|(k, v)| ((*k).into(), v.clone())).collect(),
        ))),
        _ => Err(Error::new(format!("unknown function {name}"))),
    }
}

fn test<'a>(name: &str, v: &Value<'a>, args: &[Value<'a>]) -> Result<bool> {
    Ok(match name {
        "defined" => !matches!(v, Value::Undefined),
        "undefined" => matches!(v, Value::Undefined),
        "none" => matches!(v, Value::None),
        "true" => matches!(v, Value::Bool(true)),
        "false" => matches!(v, Value::Bool(false)),
        "boolean" => matches!(v, Value::Bool(_)),
        "string" => v.as_str().is_some(),
        "mapping" => v.mapping(),
        "sequence" => v.sequence() || v.mapping(),
        "iterable" => v.iterable(),
        "number" => v.number().is_some(),
        "integer" => matches!(v, Value::Int(_) | Value::UInt(_)),
        "float" => matches!(v, Value::Float(_)),
        "sameas" => matches!((v, args.first()),
            (Value::Bool(true), Some(Value::Bool(true))) | (Value::Bool(false), Some(Value::Bool(false)))
            | (Value::None, Some(Value::None)) | (Value::Undefined, Some(Value::Undefined))),
        "equalto" | "eq" => args.first().is_some_and(|a| v.equals(a)),
        "odd" => v.integer().is_some_and(|v| v % 2 != 0),
        "even" => v.integer().is_some_and(|v| v % 2 == 0),
        _ => return Err(Error::new(format!("unknown test {name}"))),
    })
}

fn filter<'a>(
    name: &str,
    v: Value<'a>,
    args: &[Value<'a>],
    kw: &[(&str, Value<'a>)],
    track_hints: bool,
) -> Result<Value<'a>> {
    match name {
        "safe" => Ok(v),
        "default" | "d" => Ok(
            if matches!(v, Value::Undefined)
                || args.get(1).is_some_and(Value::truthy) && !v.truthy()
            {
                args.first().cloned().unwrap_or(Value::Str(""))
            } else {
                v
            },
        ),
        "length" | "count" => Ok(Value::Int(v.len() as i64)),
        "string" => Ok(if v.as_str().is_some() {
            v
        } else {
            Value::string(v.text())
        }),
        "trim" => {
            let chars = args.first().and_then(Value::as_str);
            let pred = |c: char| chars.map_or_else(|| c.is_whitespace(), |s| s.contains(c));
            if let Value::Str(s) = v {
                return Ok(Value::Str(s.trim_matches(pred)));
            }
            let text = if let Some(s) = v.as_str() {
                s
            } else {
                return Ok(Value::string(v.text().trim_matches(pred).into()));
            };
            let trimmed = text.trim_matches(pred);
            if trimmed.len() == text.len() {
                return Ok(v);
            }
            let start = trimmed.as_ptr() as usize - text.as_ptr() as usize;
            let end = start + trimmed.len();
            let (mut hints, runs) = v.sliced_annotations(start, end);
            if trimmed.is_empty() {
                hints.clear();
            }
            Ok(Value::marked_runs(trimmed.into(), hints, runs))
        }
        "capitalize" => {
            let text = v.text().to_lowercase();
            let mut chars = text.chars();
            Ok(Value::string(
                chars
                    .next()
                    .map(|c| c.to_uppercase().to_string() + chars.as_str())
                    .unwrap_or_default(),
            ))
        }
        "lower" => Ok(Value::string(v.text().to_lowercase())),
        "upper" => Ok(Value::string(v.text().to_uppercase())),
        "replace" => {
            let from = args
                .first()
                .ok_or_else(|| Error::new("replace requires two arguments"))?
                .text();
            let to = args
                .get(1)
                .ok_or_else(|| Error::new("replace requires two arguments"))?
                .text();
            let count = args.get(2).and_then(Value::integer).map(|n| n.max(0) as usize);
            let text = v.text();
            let output = match count {
                Some(count) => text.replacen(&from, &to, count),
                None => text.replace(&from, &to),
            };
            if v.hints().is_empty() || from.is_empty() {
                return Ok(Value::string(output));
            }
            let replacements: Vec<_> = text.match_indices(&from).take(count.unwrap_or(usize::MAX)).map(|(at, _)| at).collect();
            let hints = v.hints().iter().filter_map(|&end| {
                let mut position = end;
                for &at in &replacements {
                    if at < end && end < at + from.len() { return None; }
                    if at + from.len() <= end {
                        position = position + to.len() - from.len();
                    }
                }
                Some(position)
            }).collect();
            Ok(Value::marked(output, hints))
        }
        "items" => Ok(Value::list(
            v.pairs()?
                .into_iter()
                .map(|(k, v)| Value::list(vec![Value::string(k), v]))
                .collect(),
        )),
        "tojson" => {
            let kwargs = kw
                .iter()
                .map(|(k, v)| Ok(((*k).to_owned(), v.to_json()?)))
                .collect::<Result<Map<_, _>>>()?;
            let json = v.to_json()?;
            if !track_hints {
                return Ok(Value::string(
                    super::super::json::format_json(&json, &kwargs).map_err(Error::new)?,
                ));
            }
            let mut hints = Vec::new();
            v.json_hints(&json, &mut hints);
            let (text, positions) = super::super::json::format_json_hinted(&json, &kwargs, &hints)
                .map_err(Error::new)?;
            Ok(Value::marked(text, positions))
        }
        "fromjson" => Ok(Value::owned(
            serde_json::from_str(
                v.as_str()
                    .ok_or_else(|| Error::new("fromjson requires a string"))?,
            )
            .map_err(|e| Error::new(e.to_string()))?,
        )),
        "sort" => {
            let mut items = v.iter()?;
            let attr = kw
                .iter()
                .find(|(k, _)| *k == "attribute")
                .and_then(|(_, v)| v.as_str());
            let reverse = kw
                .iter()
                .find(|(k, _)| *k == "reverse")
                .map(|(_, v)| v.truthy())
                .or_else(|| args.first().map(Value::truthy))
                .unwrap_or(false);
            items.sort_by(|a, b| {
                let a = attr.map_or_else(|| a.clone(), |k| a.attr(k));
                let b = attr.map_or_else(|| b.clone(), |k| b.attr(k));
                let order = if let (Some(a), Some(b)) = (a.number(), b.number()) {
                    a.partial_cmp(&b).unwrap_or(std::cmp::Ordering::Equal)
                } else {
                    a.text().to_lowercase().cmp(&b.text().to_lowercase())
                };
                if reverse { order.reverse() } else { order }
            });
            Ok(Value::list(items))
        }
        "dictsort" => {
            if !args.is_empty() {
                return Err(Error::new("dictsort accepts only keyword arguments"));
            }
            let option = |name: &str| kw.iter().find(|(k, _)| *k == name).map(|(_, v)| v.clone());
            let case_sensitive = option("case_sensitive").is_some_and(|v| v.truthy());
            let by_value = match option("by") {
                None => false,
                Some(by) => match by.as_str() {
                    Some("key") => false,
                    Some("value") => true,
                    _ => return Err(Error::new("dictsort by must be 'key' or 'value'")),
                },
            };
            let reverse = option("reverse").is_some_and(|v| v.truthy());
            let text = |v: &Value| {
                if case_sensitive {
                    v.text()
                } else {
                    v.text().to_lowercase()
                }
            };
            let mut pairs = v.pairs()?;
            pairs.sort_by(|(ak, av), (bk, bv)| {
                let order = if by_value {
                    if let (Some(a), Some(b)) = (av.number(), bv.number()) {
                        a.partial_cmp(&b).unwrap_or(std::cmp::Ordering::Equal)
                    } else {
                        text(av).cmp(&text(bv))
                    }
                } else if case_sensitive {
                    ak.cmp(bk)
                } else {
                    ak.to_lowercase().cmp(&bk.to_lowercase())
                };
                if reverse { order.reverse() } else { order }
            });
            Ok(Value::list(
                pairs
                    .into_iter()
                    .map(|(k, v)| Value::list(vec![Value::string(k), v]))
                    .collect(),
            ))
        }
        "join" => {
            let sep = args.first().map(Value::text).unwrap_or_default();
            let mut text = String::new();
            for (i, v) in v.iter()?.iter().enumerate() {
                if i != 0 {
                    text.push_str(&sep);
                }
                v.write(&mut text);
            }
            Ok(Value::string(text))
        }
        "list" => Ok(Value::list(v.iter()?)),
        "first" => Ok(v.index(&Value::Int(0))),
        "last" => Ok(v.index(&Value::Int(-1))),
        "int" => Ok(Value::Int(
            v.integer()
                .or_else(|| v.as_str().and_then(|v| v.parse().ok()))
                .unwrap_or(0),
        )),
        _ => Err(Error::new(format!("unknown filter {name}"))),
    }
}

fn method<'a>(
    name: &str,
    v: Value<'a>,
    args: &[Value<'a>],
    kw: &[(&str, Value<'a>)],
) -> Result<Value<'a>> {
    if v.mapping() {
        return match name {
            "items" => filter("items", v, args, kw, false),
            "keys" => Ok(Value::list(v.iter()?)),
            "values" => Ok(Value::list(
                v.pairs()?.into_iter().map(|(_, v)| v).collect(),
            )),
            "get" => {
                let key = args
                    .first()
                    .ok_or_else(|| Error::new("get requires a key"))?;
                let value = v.index(key);
                Ok(if matches!(value, Value::Undefined) {
                    args.get(1).cloned().unwrap_or(Value::None)
                } else {
                    value
                })
            }
            _ => Err(Error::new(format!("unknown map method {name}"))),
        };
    }
    let text = v
        .as_str()
        .ok_or_else(|| Error::new(format!("unknown method {name}")))?;
    let argument = |i: usize| {
        args.get(i)
            .and_then(Value::as_str)
            .ok_or_else(|| Error::new(format!("{name} requires string arguments")))
    };
    match name {
        "startswith" => Ok(Value::Bool(text.starts_with(argument(0)?))),
        "endswith" => Ok(Value::Bool(text.ends_with(argument(0)?))),
        "strip" | "lstrip" | "rstrip" => {
            let chars = args.first().and_then(Value::as_str);
            let pred = |c: char| chars.map_or_else(|| c.is_whitespace(), |s| s.contains(c));
            let trimmed = match name {
                "lstrip" => text.trim_start_matches(pred),
                "rstrip" => text.trim_end_matches(pred),
                _ => text.trim_matches(pred),
            };
            let start = trimmed.as_ptr() as usize - text.as_ptr() as usize;
            let end = start + trimmed.len();
            let (mut hints, runs) = v.sliced_annotations(start, end);
            if trimmed.is_empty() {
                hints.clear();
            }
            Ok(Value::marked_runs(trimmed.into(), hints, runs))
        }
        "split" | "rsplit" => {
            let limit = args
                .get(1)
                .and_then(Value::integer)
                .filter(|n| *n >= 0)
                .map(|n| n as usize + 1)
                .unwrap_or(usize::MAX);
            let parts: Vec<_> = if args.is_empty() || matches!(args[0], Value::None) {
                text.split_whitespace()
                    .map(|s| Value::string(s.into()))
                    .collect()
            } else {
                let sep = argument(0)?;
                if sep.is_empty() {
                    return Err(Error::new("empty separator"));
                }
                if name == "rsplit" {
                    let mut items: Vec<_> = text
                        .rsplitn(limit, sep)
                        .map(|s| Value::string(s.into()))
                        .collect();
                    items.reverse();
                    items
                } else {
                    text.splitn(limit, sep)
                        .map(|s| Value::string(s.into()))
                        .collect()
                }
            };
            Ok(Value::list(parts))
        }
        "replace" | "lower" | "upper" | "capitalize" => filter(name, v, args, kw, false),
        _ => Err(Error::new(format!("unknown string method {name}"))),
    }
}
