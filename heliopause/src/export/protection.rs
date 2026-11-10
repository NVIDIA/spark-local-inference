/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Publisher signatures and independently compressed executable resources.
use pyo3::{exceptions::PyValueError, prelude::*};
use super::{ExportResult as Result, invalid, wire};
use ring::{
    digest,
    signature::{self, KeyPair},
};
use serde::{Deserialize, Serialize};
use std::{
    fs,
    io::{Read, Write},
    path::Path,
};

pub(crate) fn sha256(bytes: &[u8]) -> Vec<u8> {
    digest::digest(&digest::SHA256, bytes).as_ref().to_vec()
}
pub(crate) fn fields(values: &[&[u8]]) -> Vec<u8> {
    let mut result = Vec::new();
    for value in values {
        result.extend_from_slice(&(value.len() as u64).to_le_bytes());
        result.extend_from_slice(value);
    }
    result
}
pub(crate) fn signature_message(core: &[u8], signer: &[u8], kind: u32, body: &[u8]) -> Vec<u8> {
    // Match the authentication domain used by every compatible signed-core reader.
    fields(&[
        b"flare-package-signature-v6",
        &6u32.to_le_bytes(),
        core,
        signer,
        &kind.to_le_bytes(),
        body,
    ])
}
pub(crate) fn validate_namespace(value: &str) -> Result<()> {
    if value.is_empty()
        || value.len() > 96
        || !value.as_bytes()[0].is_ascii_alphanumeric()
        || !value
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'_' | b'-'))
    {
        return Err(invalid(
            "publisher namespace must be a Hugging Face owner name",
        ));
    }
    Ok(())
}
pub(crate) fn validate_source(namespace: &str, source: &str) -> Result<()> {
    validate_namespace(namespace)?;
    let repository = source
        .strip_prefix(&format!("https://huggingface.co/{namespace}/"))
        .ok_or_else(|| {
            invalid("official_source must be https://huggingface.co/<publisher>/<repository>")
        })?;
    if repository.is_empty()
        || repository.len() > 96
        || !repository.as_bytes()[0].is_ascii_alphanumeric()
        || !repository
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'_' | b'-'))
    {
        return Err(invalid(
            "official_source must be https://huggingface.co/<publisher>/<repository>",
        ));
    }
    Ok(())
}
pub(crate) fn validate_license(value: &wire::License) -> Result<()> {
    if value.name.is_empty()
        || value.name.chars().count() > 128
        || value.name.chars().any(char::is_control)
    {
        return Err(invalid(
            "license name must be 1 to 128 printable characters",
        ));
    }
    if !value.link.is_empty()
        && (!value.link.starts_with("https://")
            || value.link.len() <= 8
            || value.link.len() > 512
            || !value.link.bytes().all(|b| b.is_ascii_graphic()))
    {
        return Err(invalid(
            "license link must be an https URL of printable ASCII, at most 512 characters",
        ));
    }
    Ok(())
}

pub(crate) struct Publisher {
    key: signature::Ed25519KeyPair,
    pub namespace: String,
    pub official_source: String,
    pub development: bool,
    pub license: Option<wire::License>,
}
impl Publisher {
    /// Signing key retained by this publisher for payload-format publication.
    pub(crate) fn signing_key(&self) -> &signature::Ed25519KeyPair {
        &self.key
    }

    pub(crate) fn from_seed(
        seed: &[u8],
        namespace: String,
        official_source: String,
        development: bool,
        license: Option<wire::License>,
    ) -> Result<Self> {
        if development {
            if namespace != "local-development" || official_source != "local-development" {
                return Err(invalid("invalid local development publisher"));
            }
        } else {
            validate_source(&namespace, &official_source)?;
        }
        if let Some(value) = &license {
            validate_license(value)?;
        }
        let key = signature::Ed25519KeyPair::from_seed_unchecked(seed)
            .map_err(|_| invalid("publishing requires an Ed25519 private key"))?;
        Ok(Self {
            key,
            namespace,
            official_source,
            development,
            license,
        })
    }
    pub(crate) fn load(
        path: &Path,
        password: Option<&[u8]>,
        namespace: String,
        official_source: String,
        license: Option<wire::License>,
    ) -> Result<Self> {
        let mut file = fs::File::open(path)?;
        let metadata = file.metadata()?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::MetadataExt;
            if metadata.uid() != unsafe { libc::getuid() } || metadata.mode() & 0o077 != 0 {
                return Err(invalid(format!(
                    "signing key must have owner-only permissions: {}",
                    path.display()
                )));
            }
        }
        #[cfg(windows)]
        validate_private_file(path)?;
        if !metadata.is_file() {
            return Err(invalid("signing key must be an owner-only regular file"));
        }
        if metadata.len() > 16384 {
            return Err(invalid("signing key exceeds 16 KiB"));
        }
        let mut encoded = Vec::new();
        std::io::Read::by_ref(&mut file)
            .take(16385)
            .read_to_end(&mut encoded)?;
        if encoded.len() > 16384 {
            return Err(invalid("signing key exceeds 16 KiB"));
        }
        let mut seed = load_key_seed(&encoded, password)?;
        let result = Self::from_seed(
            &seed,
            namespace.clone(),
            official_source.clone(),
            namespace == "local-development" && official_source == "local-development",
            license,
        );
        seed.fill(0);
        encoded.fill(0);
        result
    }
    pub(crate) fn public_key(&self) -> &[u8] {
        self.key.public_key().as_ref()
    }
    pub(crate) fn fingerprint(&self) -> Vec<u8> {
        sha256(self.public_key())
    }
}
#[derive(Clone, Debug, Deserialize, Serialize)]
pub(crate) struct PublisherRequest {
    pub namespace: String,
    pub official_source: String,
    #[serde(default)]
    pub development: bool,
    pub license: Option<wire::License>,
    pub key_path: Option<std::path::PathBuf>,
    #[serde(default)]
    pub password: Option<Vec<u8>>,
    #[serde(default)]
    pub seed: Option<Vec<u8>>,
    #[serde(default)]
    pub expected_public_key: Option<Vec<u8>>,
}
impl PublisherRequest {
    pub(crate) fn publisher(&self) -> Result<Publisher> {
        if let Some(path) = &self.key_path {
            if self.development {
                if self.namespace != "local-development"
                    || self.official_source != "local-development"
                {
                    return Err(invalid("invalid local development publisher"));
                }
            } else {
                validate_source(&self.namespace, &self.official_source)?;
            }
            {
                let mut result = Publisher::load(
                    path,
                    self.password.as_deref(),
                    self.namespace.clone(),
                    self.official_source.clone(),
                    self.license.clone(),
                )?;
                if self
                    .expected_public_key
                    .as_deref()
                    .is_none_or(|expected| expected != result.public_key())
                {
                    return Err(invalid(
                        "signing key changed since publisher initialization",
                    ));
                }
                result.development = self.development;
                if self.development
                    && (result.namespace != "local-development"
                        || result.official_source != "local-development")
                {
                    return Err(invalid("invalid local development publisher"));
                }
                Ok(result)
            }
        } else {
            Publisher::from_seed(
                self.seed
                    .as_deref()
                    .ok_or_else(|| invalid("publisher private key is missing"))?,
                self.namespace.clone(),
                self.official_source.clone(),
                self.development,
                self.license.clone(),
            )
        }
    }
}

fn publisher_info(publisher: &Publisher) -> serde_json::Value {
    serde_json::json!({"public_key":publisher.public_key(),"fingerprint":publisher.fingerprint()})
}
#[pyfunction]
#[pyo3(signature=(path,password=None,namespace="local-development",official_source="local-development"))]
fn export_key_info(
    py: Python<'_>,
    path: String,
    password: Option<Vec<u8>>,
    namespace: &str,
    official_source: &str,
) -> PyResult<String> {
    let namespace = namespace.to_owned();
    let official_source = official_source.to_owned();
    py.detach(|| {
        let path = Path::new(&path);
        let file = fs::File::open(path)?;
        let metadata = file.metadata()?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::MetadataExt;
            if metadata.uid() != unsafe { libc::getuid() } || metadata.mode() & 0o077 != 0 {
                return Err(invalid(format!(
                    "signing key must have owner-only permissions: {}",
                    path.display()
                )));
            }
        }
        if !metadata.is_file() || metadata.len() > 16384 {
            return Err(invalid(
                "signing key must be a regular file no larger than 16 KiB",
            ));
        }
        #[cfg(windows)]
        validate_private_file(path)?;
        let mut prefix = [0; 37];
        let count = Read::read(&mut file.take(37), &mut prefix)?;
        if password.is_none()
            && prefix[..count].starts_with(b"-----BEGIN ENCRYPTED PRIVATE KEY-----")
        {
            return Ok(serde_json::to_string(
                &serde_json::json!({"encrypted":true}),
            )?);
        }
        let publisher =
            Publisher::load(path, password.as_deref(), namespace, official_source, None)?;
        Ok::<_, Box<dyn std::error::Error + Send + Sync>>(serde_json::to_string(&publisher_info(
            &publisher,
        ))?)
    })
    .map_err(|e| PyValueError::new_err(e.to_string()))
}
#[pyfunction]
fn export_development_publisher(py: Python<'_>, directory: String) -> PyResult<String> {
    py.detach(|| {
        development_publisher(Path::new(&directory))
            .and_then(|value| Ok(serde_json::to_string(&value)?))
    })
    .map_err(|e| PyValueError::new_err(e.to_string()))
}
fn development_publisher(directory: &Path) -> Result<serde_json::Value> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::DirBuilderExt;
        fs::DirBuilder::new()
            .recursive(true)
            .mode(0o700)
            .create(directory)?;
    }
    #[cfg(not(unix))]
    fs::create_dir_all(directory)?;
    let path = directory.join("development-signing-key.pem");
    if !path.try_exists()? {
        let key = openssl::pkey::PKey::generate_ed25519()?;
        let mut encoded = key.private_key_to_pem_pkcs8()?;
        let mut temporary = tempfile::Builder::new()
            .prefix(".development-key-")
            .tempfile_in(directory)?;
        #[cfg(windows)]
        restrict_private_file(temporary.path())?;
        temporary.write_all(&encoded)?;
        temporary.as_file().sync_all()?;
        encoded.fill(0);
        match fs::hard_link(temporary.path(), &path) {
            Ok(()) => (),
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => (),
            Err(e) => return Err(e.into()),
        }
    }
    let publisher = Publisher::load(
        &path,
        None,
        "local-development".into(),
        "local-development".into(),
        None,
    )?;
    let profile = serde_json::json!([{"name":"Local development signer","namespace":"local-development","public_key":publisher.public_key(),"development":true}]);
    let mut temporary = tempfile::Builder::new()
        .prefix(".development-profile-")
        .tempfile_in(directory)?;
    temporary.write_all(serde_json::to_string_pretty(&profile)?.as_bytes())?;
    temporary.write_all(b"\n")?;
    temporary.as_file().sync_all()?;
    temporary.persist(directory.join("development-publisher.json"))?;
    super::archive::sync_directory(directory)?;
    Ok(
        serde_json::json!({"key_path":path,"public_key":publisher.public_key(),"fingerprint":publisher.fingerprint()}),
    )
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(export_key_info, module)?)?;
    module.add_function(wrap_pyfunction!(export_development_publisher, module)?)?;
    Ok(())
}

#[cfg(windows)]
fn restrict_private_file(path: &Path) -> Result<()> {
    use std::{os::windows::ffi::OsStrExt, ptr};
    use windows_sys::Win32::{
        Foundation::LocalFree,
        Security::{
            Authorization::{
                ConvertStringSecurityDescriptorToSecurityDescriptorW, SDDL_REVISION_1,
            },
            DACL_SECURITY_INFORMATION, PROTECTED_DACL_SECURITY_INFORMATION, SetFileSecurityW,
        },
    };
    let text: Vec<u16> = "D:P(A;;FA;;;SY)(A;;FA;;;OW)\0".encode_utf16().collect();
    let path: Vec<u16> = path.as_os_str().encode_wide().chain(Some(0)).collect();
    let mut descriptor = ptr::null_mut();
    unsafe {
        if ConvertStringSecurityDescriptorToSecurityDescriptorW(
            text.as_ptr(),
            SDDL_REVISION_1,
            &mut descriptor,
            ptr::null_mut(),
        ) == 0
        {
            return Err(std::io::Error::last_os_error().into());
        }
        let result = SetFileSecurityW(
            path.as_ptr(),
            DACL_SECURITY_INFORMATION | PROTECTED_DACL_SECURITY_INFORMATION,
            descriptor,
        );
        LocalFree(descriptor);
        if result == 0 {
            return Err(std::io::Error::last_os_error().into());
        }
    }
    Ok(())
}
#[cfg(windows)]
fn validate_private_file(path: &Path) -> Result<()> {
    use std::{os::windows::ffi::OsStrExt, ptr};
    use windows_sys::Win32::{
        Foundation::LocalFree,
        Security::{
            ACL,
            Authorization::{ConvertStringSidToSidW, GetNamedSecurityInfoW, SE_FILE_OBJECT},
            DACL_SECURITY_INFORMATION, EqualSid, GetAce, OWNER_SECURITY_INFORMATION,
        },
    };
    let path: Vec<u16> = path.as_os_str().encode_wide().chain(Some(0)).collect();
    let mut owner = ptr::null_mut();
    let mut acl: *mut ACL = ptr::null_mut();
    let mut descriptor = ptr::null_mut();
    let mut allowed = Vec::new();
    unsafe {
        let status = GetNamedSecurityInfoW(
            path.as_ptr(),
            SE_FILE_OBJECT,
            OWNER_SECURITY_INFORMATION | DACL_SECURITY_INFORMATION,
            &mut owner,
            ptr::null_mut(),
            &mut acl,
            ptr::null_mut(),
            &mut descriptor,
        );
        if status != 0 {
            return Err(std::io::Error::from_raw_os_error(status as i32).into());
        }
        let result = (|| -> Result<()> {
            if owner.is_null() || acl.is_null() {
                return Err(invalid("signing key must have owner-only permissions"));
            }
            for text in ["S-1-3-4\0", "S-1-5-18\0"] {
                let text: Vec<u16> = text.encode_utf16().collect();
                let mut sid = ptr::null_mut();
                if ConvertStringSidToSidW(text.as_ptr(), &mut sid) == 0 {
                    return Err(std::io::Error::last_os_error().into());
                }
                allowed.push(sid);
            }
            for i in 0..(*acl).AceCount {
                let mut ace: *mut std::ffi::c_void = ptr::null_mut();
                if GetAce(acl, i as u32, &mut ace) == 0 {
                    return Err(std::io::Error::last_os_error().into());
                }
                let bytes = ace.cast::<u8>();
                let kind = *bytes;
                if kind == 1 {
                    continue;
                }
                let sid = bytes.add(8).cast();
                if kind != 0
                    || !std::iter::once(owner)
                        .chain(allowed.iter().copied())
                        .any(|allowed| EqualSid(sid, allowed) != 0)
                {
                    return Err(invalid("signing key must have owner-only permissions"));
                }
            }
            Ok(())
        })();
        for sid in allowed {
            LocalFree(sid);
        }
        LocalFree(descriptor);
        result
    }
}

#[cfg(not(openssl3))]
fn load_key_seed(encoded: &[u8], password: Option<&[u8]>) -> Result<Vec<u8>> {
    let key = match password {
        Some(password) => openssl::pkey::PKey::private_key_from_pem_passphrase(encoded, password),
        None => openssl::pkey::PKey::private_key_from_pem(encoded),
    }?;
    if key.id() != openssl::pkey::Id::ED25519 {
        return Err(invalid("publishing requires an Ed25519 private key"));
    }
    Ok(key.raw_private_key()?)
}

#[cfg(openssl3)]
fn load_key_seed(encoded: &[u8], password: Option<&[u8]>) -> Result<Vec<u8>> {
    use openssl_sys as ffi;
    use std::ptr;
    struct Import {
        context: *mut ffi::OSSL_LIB_CTX,
        default: *mut ffi::OSSL_PROVIDER,
        legacy: *mut ffi::OSSL_PROVIDER,
        decoder: *mut ffi::OSSL_DECODER_CTX,
        key: *mut ffi::EVP_PKEY,
    }
    impl Drop for Import {
        fn drop(&mut self) {
            unsafe {
                if !self.key.is_null() {
                    ffi::EVP_PKEY_free(self.key);
                }
                if !self.decoder.is_null() {
                    ffi::OSSL_DECODER_CTX_free(self.decoder);
                }
                if !self.legacy.is_null() {
                    ffi::OSSL_PROVIDER_unload(self.legacy);
                }
                if !self.default.is_null() {
                    ffi::OSSL_PROVIDER_unload(self.default);
                }
                if !self.context.is_null() {
                    ffi::OSSL_LIB_CTX_free(self.context);
                }
            }
        }
    }
    let mut import = Import {
        context: ptr::null_mut(),
        default: ptr::null_mut(),
        legacy: ptr::null_mut(),
        decoder: ptr::null_mut(),
        key: ptr::null_mut(),
    };
    let error = || {
        invalid(format!(
            "import Ed25519 PKCS#8 signing key: {}",
            openssl::error::ErrorStack::get()
        ))
    };
    // The private context keeps legacy key algorithms isolated from other OpenSSL callers.
    unsafe {
        import.context = ffi::OSSL_LIB_CTX_new();
        if import.context.is_null() {
            return Err(error());
        }
        import.default = ffi::OSSL_PROVIDER_load(import.context, c"default".as_ptr());
        if import.default.is_null() {
            return Err(error());
        }
        import.legacy = ffi::OSSL_PROVIDER_load(import.context, c"legacy".as_ptr());
        if import.legacy.is_null() {
            return Err(error());
        }
        import.decoder = ffi::OSSL_DECODER_CTX_new_for_pkey(
            &mut import.key,
            c"PEM".as_ptr(),
            ptr::null(),
            c"ED25519".as_ptr(),
            ffi::EVP_PKEY_KEYPAIR,
            import.context,
            ptr::null(),
        );
        if import.decoder.is_null() {
            return Err(error());
        }
        let password = password.unwrap_or(&[]);
        if ffi::OSSL_DECODER_CTX_set_passphrase(import.decoder, password.as_ptr(), password.len())
            != 1
        {
            return Err(error());
        }
        let mut data = encoded.as_ptr();
        let mut length = encoded.len();
        if ffi::OSSL_DECODER_from_data(import.decoder, &mut data, &mut length) != 1
            || import.key.is_null()
        {
            return Err(error());
        }
        if ffi::EVP_PKEY_id(import.key) != ffi::EVP_PKEY_ED25519 {
            return Err(invalid("publishing requires an Ed25519 private key"));
        }
        let mut seed = vec![0; 32];
        let mut length = seed.len();
        if ffi::EVP_PKEY_get_raw_private_key(import.key, seed.as_mut_ptr(), &mut length) != 1
            || length != 32
        {
            seed.fill(0);
            return Err(error());
        }
        Ok(seed)
    }
}

#[cfg(all(test, openssl3))]
mod key_context_tests {
    use super::*;
    #[cfg(unix)]
    #[test]
    fn insecure_signing_key_reports_owner_only_permissions() {
        use std::os::unix::fs::PermissionsExt;
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("signing-key.pem");
        let pem = openssl::pkey::PKey::generate_ed25519()
            .unwrap()
            .private_key_to_pem_pkcs8()
            .unwrap();
        fs::write(&path, &pem).unwrap();
        fs::set_permissions(&path, fs::Permissions::from_mode(0o644)).unwrap();
        let load = || {
            Publisher::load(
                &path,
                None,
                "local-development".into(),
                "local-development".into(),
                None,
            )
        };
        let error = load().err().expect("insecure signing key must be rejected");
        assert!(error.to_string().contains("owner-only permissions"));
        assert_eq!(fs::read(&path).unwrap(), pem);
        fs::set_permissions(&path, fs::Permissions::from_mode(0o600)).unwrap();
        assert!(load().is_ok());
    }
    fn global_des_available() -> bool {
        unsafe {
            let cipher = openssl_sys::EVP_CIPHER_fetch(
                std::ptr::null_mut(),
                c"DES-CBC".as_ptr(),
                std::ptr::null(),
            );
            let available = !cipher.is_null();
            if available {
                openssl_sys::EVP_CIPHER_free(cipher);
            }
            openssl::error::ErrorStack::get();
            available
        }
    }
    #[test]
    fn key_import_preserves_the_callers_default_algorithm_context() {
        openssl::init();
        let key = openssl::pkey::PKey::generate_ed25519().unwrap();
        let pem = key
            .private_key_to_pem_pkcs8_passphrase(
                openssl::symm::Cipher::aes_256_cbc(),
                b"fixture-password",
            )
            .unwrap();
        let before = global_des_available();
        let mut seed = load_key_seed(&pem, Some(b"fixture-password")).unwrap();
        assert_eq!(seed, key.raw_private_key().unwrap());
        seed.fill(0);
        assert_eq!(global_des_available(), before);
    }
}
