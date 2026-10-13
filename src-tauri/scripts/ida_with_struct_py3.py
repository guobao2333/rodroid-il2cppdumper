# -*- coding: utf-8 -*-
import json
import idaapi
import idc
import ida_funcs
import ida_bytes
import ida_segment
import ida_typeinf
import ida_dirtree
import ida_xref
import ida_auto

processFields = [
	"ScriptMethod",
	"ScriptString",
	"ScriptMetadata",
	"ScriptMetadataMethod",
	"Addresses",
	"TypeInfoPointers",
	"TypeRefPointers",
	"FieldInfos",
]

imageBase = idaapi.get_imagebase()

def get_addr(addr):
	return imageBase + addr

def set_name(addr, name):
	ret = idc.set_name(addr, name, idc.SN_NOWARN | idc.SN_NOCHECK)
	if ret == 0:
		new_name = name + '_' + str(addr)
		ret = idc.set_name(addr, new_name, idc.SN_NOWARN | idc.SN_NOCHECK)

def make_function(start, end):
	next_func = idc.get_next_func(start)
	if next_func < end:
		end = next_func
	if idc.get_func_attr(start, idc.FUNCATTR_START) == start:
		ida_funcs.del_func(start)
	ida_funcs.add_func(start, end)

def create_fake_string_segment(start_addr, end_addr):
	ida_segment.add_segm(0, start_addr, end_addr, ".fake_strings", "DATA")

def add_cross_reference(from_addr, to_addr):
	ida_xref.add_dref(from_addr, to_addr, ida_xref.XREF_USER | ida_xref.dr_I)

# Start execution
ida_auto.set_ida_state(ida_auto.st_Work) # Disable auto-analysis during script execution for speed
idc.set_inf_attr(idc.INF_SHORT_DEMNAMES, idc.DEMNAM_GCC3)

path = idaapi.ask_file(False, '*.json', 'script.json from Il2cppdumper')
hpath = idaapi.ask_file(False, '*.h', 'il2cpp.h from Il2cppdumper')
ida_typeinf.parse_decls(None, open(hpath, 'r').read(), None, 0)
data = json.loads(open(path, 'rb').read().decode('utf-8'))

# Setup DirTree for IDA 9.3+ Folders
func_dirtree = None
try:
	func_dirtree = ida_dirtree.dirtree_t(ida_dirtree.DIRTREE_FUNCS)
except:
	pass

if "Addresses" in data and "Addresses" in processFields:
	idaapi.show_wait_box("Processing Addresses...")
	addresses = data["Addresses"]
	for index in range(len(addresses) - 1):
		start = get_addr(addresses[index])
		end = get_addr(addresses[index + 1])
		make_function(start, end)
	idaapi.hide_wait_box()

if "ScriptMethod" in data and "ScriptMethod" in processFields:
	idaapi.show_wait_box("Processing Methods...")
	scriptMethods = data["ScriptMethod"]
	for scriptMethod in scriptMethods:
		addr = get_addr(scriptMethod["Address"])
		name = scriptMethod["Name"]
		set_name(addr, name)
		
		# Demangle friendly folders
		if func_dirtree and "Group" in scriptMethod and scriptMethod["Group"]:
			group = scriptMethod["Group"]
			func_dirtree.mkdir(group)
			func_name = ida_funcs.get_func_name(addr)
			func_dirtree.rename(func_name, "{}/{}".format(group, func_name))
			
		signature = scriptMethod["Signature"]
		pt = ida_typeinf.tinfo_t()
		if ida_typeinf.parse_decl(pt, None, signature, 0):
			ida_typeinf.apply_tinfo(addr, pt, ida_typeinf.TINFO_DEFINITE)
	idaapi.hide_wait_box()

if "ScriptString" in data and "ScriptString" in processFields:
	idaapi.show_wait_box("Processing Strings...")
	scriptStrings = data["ScriptString"]
	
	# Calculate size needed
	total_len = 0
	for scriptString in scriptStrings:
		value = scriptString["Value"].encode('utf-8')
		total_len += len(value) + 1
	
	fake_seg_start = idaapi.get_inf_structure().max_ea
	# Pad to 0x1000 alignment
	fake_seg_start = (fake_seg_start + 0xFFF) & ~0xFFF
	create_fake_string_segment(fake_seg_start, fake_seg_start + total_len)
	
	current_fake_addr = fake_seg_start
	
	# Cache type for strings
	str_type = ida_typeinf.tinfo_t()
	ida_typeinf.parse_decl(str_type, None, "const char* const;", 0)
	
	for scriptString in scriptStrings:
		addr = get_addr(scriptString["Address"])
		value = scriptString["Value"].encode('utf-8')
		
		# Write bytes to fake segment
		ida_bytes.put_bytes(current_fake_addr, value + b'\x00')
		# Patch original literal
		if idaapi.get_inf_structure().is_64bit():
			ida_bytes.patch_qword(addr, current_fake_addr)
		else:
			ida_bytes.patch_dword(addr, current_fake_addr)
		
		ida_typeinf.apply_tinfo(addr, str_type, ida_typeinf.TINFO_DEFINITE)
		
		name = scriptString.get("Name")
		if not name:
			name = "StringLiteral_" + str(scriptString["Address"])
		
		idc.set_name(addr, name, idc.SN_NOWARN | idc.SN_NOCHECK)
		current_fake_addr += len(value) + 1
	idaapi.hide_wait_box()

if "ScriptMetadata" in data and "ScriptMetadata" in processFields:
	idaapi.show_wait_box("Processing Metadata...")
	scriptMetadatas = data["ScriptMetadata"]
	for scriptMetadata in scriptMetadatas:
		addr = get_addr(scriptMetadata["Address"])
		name = scriptMetadata["Name"]
		set_name(addr, name)
		idc.set_cmt(addr, name, 1)
		
		if "Signature" in scriptMetadata and scriptMetadata["Signature"]:
			signature = scriptMetadata["Signature"]
			pt = ida_typeinf.tinfo_t()
			if ida_typeinf.parse_decl(pt, None, signature + ";", 0):
				ida_typeinf.apply_tinfo(addr, pt, ida_typeinf.TINFO_DEFINITE)
	idaapi.hide_wait_box()

if "ScriptMetadataMethod" in data and "ScriptMetadataMethod" in processFields:
	idaapi.show_wait_box("Processing MethodInfo Cross-References...")
	scriptMetadataMethods = data["ScriptMetadataMethod"]
	for scriptMetadataMethod in scriptMetadataMethods:
		addr = get_addr(scriptMetadataMethod["Address"])
		name = scriptMetadataMethod["Name"]
		methodAddr = get_addr(scriptMetadataMethod["MethodAddress"])
		set_name(addr, name)
		
		if methodAddr > imageBase:
			add_cross_reference(methodAddr, addr)
	idaapi.hide_wait_box()

if "TypeInfoPointers" in data and "TypeInfoPointers" in processFields:
	idaapi.show_wait_box("Processing TypeInfo Pointers...")
	for typeInfo in data["TypeInfoPointers"]:
		addr = get_addr(typeInfo["Address"])
		set_name(addr, typeInfo["Name"])
		
		type_str = typeInfo.get("Type")
		if type_str:
			pt = ida_typeinf.tinfo_t()
			if ida_typeinf.parse_decl(pt, None, type_str + ";", 0):
				ida_typeinf.apply_tinfo(addr, pt, ida_typeinf.TINFO_DEFINITE)
	idaapi.hide_wait_box()

if "FieldInfos" in data and "FieldInfos" in processFields:
	idaapi.show_wait_box("Processing Field Infos...")
	for fieldInfo in data["FieldInfos"]:
		addr = get_addr(fieldInfo["Address"])
		set_name(addr, fieldInfo["Name"])
		idc.set_cmt(addr, fieldInfo["Value"], 1)
	idaapi.hide_wait_box()

ida_auto.set_ida_state(ida_auto.st_Ready) # Re-enable auto-analysis
print('Il2CppDumper Script finished successfully!')
