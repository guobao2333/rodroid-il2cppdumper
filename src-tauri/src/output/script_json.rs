use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScriptMethod {
    #[serde(rename = "Address")]
    pub address: u64,
    #[serde(rename = "Name")]
    pub name: String,
    #[serde(rename = "Signature")]
    pub signature: String,
    #[serde(rename = "TypeSignature")]
    pub type_signature: String,
    #[serde(rename = "DotNetSignature", skip_serializing_if = "Option::is_none")]
    pub dotnet_signature: Option<String>,
    #[serde(rename = "Group", skip_serializing_if = "Option::is_none")]
    pub group: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScriptString {
    #[serde(rename = "Address")]
    pub address: u64,
    #[serde(rename = "Value")]
    pub value: String,
    #[serde(rename = "Name", skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScriptMetadata {
    #[serde(rename = "Address")]
    pub address: u64,
    #[serde(rename = "Name")]
    pub name: String,
    #[serde(rename = "Signature", skip_serializing_if = "Option::is_none")]
    pub signature: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScriptMetadataMethod {
    #[serde(rename = "Address")]
    pub address: u64,
    #[serde(rename = "Name")]
    pub name: String,
    #[serde(rename = "MethodAddress")]
    pub method_address: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScriptTypeInfo {
    #[serde(rename = "Address")]
    pub address: u64,
    #[serde(rename = "Name")]
    pub name: String,
    #[serde(rename = "Type")]
    pub type_str: String,
    #[serde(rename = "DotNetType", skip_serializing_if = "Option::is_none")]
    pub dotnet_type: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScriptFieldInfo {
    #[serde(rename = "Address")]
    pub address: u64,
    #[serde(rename = "Name")]
    pub name: String,
    #[serde(rename = "Value")]
    pub value: String,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ScriptJson {
    #[serde(rename = "ScriptMethod")]
    pub script_methods: Vec<ScriptMethod>,
    #[serde(rename = "ScriptString")]
    pub script_strings: Vec<ScriptString>,
    #[serde(rename = "ScriptMetadata")]
    pub script_metadata: Vec<ScriptMetadata>,
    #[serde(rename = "ScriptMetadataMethod")]
    pub script_metadata_methods: Vec<ScriptMetadataMethod>,
    #[serde(rename = "Addresses")]
    pub addresses: Vec<u64>,
    #[serde(rename = "TypeInfoPointers", skip_serializing_if = "Vec::is_empty")]
    pub type_info_pointers: Vec<ScriptTypeInfo>,
    #[serde(rename = "TypeRefPointers", skip_serializing_if = "Vec::is_empty")]
    pub type_ref_pointers: Vec<ScriptTypeInfo>,
    #[serde(rename = "FieldInfos", skip_serializing_if = "Vec::is_empty")]
    pub field_infos: Vec<ScriptFieldInfo>,
    #[serde(rename = "FieldRvas", skip_serializing_if = "Vec::is_empty")]
    pub field_rvas: Vec<ScriptFieldInfo>,
}

impl ScriptJson {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StringLiteralEntry {
    pub index: usize,
    pub value: String,
}
