from pathlib import Path
from typing import List, Optional

import tree_sitter_go as tsg
from tree_sitter import Language, Parser

from src.models.chunk import Chunk, ChunkType, generate_id

GO_LANGUAGE = Language(tsg.language())


def _is_test_file(file_path: str, tree, source: bytes) -> bool:
    """
    Determine if a Go file is a test file.

    A file is considered a test file if:
    1. The file path ends with `_test.go`, OR
    2. The file imports the `"testing"` package
    """
    if file_path.endswith("_test.go"):
        return True

    # Check for import of "testing" package
    root = tree.root_node
    for child in root.children:
        if child.type == "import_declaration":
            for node in child.children:
                if node.type == "import_spec":
                    # Single import: import "testing"
                    for spec_child in node.children:
                        if spec_child.type == "interpreted_string_literal":
                            import_path = source[spec_child.start_byte:spec_child.end_byte].decode("utf-8")
                            if import_path == '"testing"':
                                return True
                elif node.type == "import_spec_list":
                    # Grouped imports: import ( "testing" ... )
                    for spec in node.children:
                        if spec.type == "import_spec":
                            for spec_child in spec.children:
                                if spec_child.type == "interpreted_string_literal":
                                    import_path = source[spec_child.start_byte:spec_child.end_byte].decode("utf-8")
                                    if import_path == '"testing"':
                                        return True

    return False


def parse_file(file_path: str) -> List[Chunk]:
    """Parse a Go source file and return all semantic chunks."""
    source = Path(file_path).read_bytes()
    parser = Parser(GO_LANGUAGE)
    tree = parser.parse(source)

    is_test = _is_test_file(file_path, tree, source)

    package_chunk = _extract_package(tree, source, file_path, is_test)
    package_name = package_chunk.name if package_chunk else ""

    struct_chunks = _extract_structs(tree, source, file_path, package_name, is_test)
    method_chunks = _extract_methods(tree, source, file_path, package_name, struct_chunks, is_test)

    chunks: List[Chunk] = []
    if package_chunk:
        chunks.append(package_chunk)
    chunks.extend(struct_chunks)
    chunks.extend(method_chunks)
    chunks.extend(_extract_functions(tree, source, file_path, package_name, is_test))
    chunks.extend(_extract_interfaces(tree, source, file_path, package_name, is_test))
    chunks.extend(_extract_consts(tree, source, file_path, package_name, is_test))
    chunks.extend(_extract_vars(tree, source, file_path, package_name, is_test))
    chunks.extend(_extract_blocks(tree, source, file_path, package_name, is_test))
    chunks.extend(_extract_type_aliases(tree, source, file_path, package_name, is_test))

    return chunks


def _extract_package(tree, source: bytes, file_path: str, is_test: bool) -> Optional[Chunk]:
    """Extract the package declaration chunk."""
    root = tree.root_node

    package_clause = None
    package_idx = -1
    for i, child in enumerate(root.children):
        if child.type == "package_clause":
            package_clause = child
            package_idx = i
            break

    if package_clause is None:
        return None

    # Collect all consecutive comment nodes immediately before the package clause
    doc_nodes = []
    for i in range(package_idx - 1, -1, -1):
        if root.children[i].type == "comment":
            doc_nodes.insert(0, root.children[i])
        else:
            break

    name = ""
    for child in package_clause.children:
        if child.type == "package_identifier":
            name = source[child.start_byte:child.end_byte].decode("utf-8")
            break

    if doc_nodes:
        start_node = doc_nodes[0]
        doc = source[start_node.start_byte:doc_nodes[-1].end_byte].decode("utf-8")
    else:
        start_node = package_clause
        doc = ""

    content = source[start_node.start_byte:package_clause.end_byte].decode("utf-8")

    return Chunk(
        id=generate_id(content),
        type=ChunkType.PACKAGE,
        content=content,
        name=name,
        package_name=name,
        file_path=file_path,
        start_line=start_node.start_point[0] + 1,
        end_line=package_clause.end_point[0] + 1,
        doc=doc,
        signature="",
        is_test=is_test,
        low_quality=False,
    )


def _extract_functions(tree, source: bytes, file_path: str, package_name: str, is_test: bool) -> List[Chunk]:
    """Extract standalone function chunks (no receiver)."""
    root = tree.root_node
    children = root.children
    chunks = []

    for i, child in enumerate(children):
        if child.type != "function_declaration":
            continue

        doc_nodes = []
        for j in range(i - 1, -1, -1):
            if children[j].type == "comment":
                doc_nodes.insert(0, children[j])
            else:
                break

        name_node = next((c for c in child.children if c.type == "identifier"), None)
        block_node = next((c for c in child.children if c.type == "block"), None)
        if name_node is None or block_node is None:
            continue

        name = source[name_node.start_byte:name_node.end_byte].decode("utf-8")
        signature = source[child.start_byte:block_node.start_byte].decode("utf-8").rstrip()

        if doc_nodes:
            start_node = doc_nodes[0]
            doc = source[start_node.start_byte:doc_nodes[-1].end_byte].decode("utf-8")
        else:
            start_node = child
            doc = ""

        content = source[start_node.start_byte:child.end_byte].decode("utf-8")

        chunks.append(Chunk(
            id=generate_id(content),
            type=ChunkType.FUNCTION,
            content=content,
            name=name,
            package_name=package_name,
            file_path=file_path,
            start_line=start_node.start_point[0] + 1,
            end_line=child.end_point[0] + 1,
            doc=doc,
            signature=signature,
            is_test=is_test,
            low_quality=False,
        ))

    return chunks


def _extract_structs(tree, source: bytes, file_path: str, package_name: str, is_test: bool) -> List[Chunk]:
    """Extract struct type declaration chunks."""
    root = tree.root_node
    children = root.children
    chunks = []

    for i, child in enumerate(children):
        if child.type != "type_declaration":
            continue
        type_spec = next((c for c in child.children if c.type == "type_spec"), None)
        if type_spec is None or not any(c.type == "struct_type" for c in type_spec.children):
            continue

        doc_nodes = []
        for j in range(i - 1, -1, -1):
            if children[j].type == "comment":
                doc_nodes.insert(0, children[j])
            else:
                break

        name_node = next(c for c in type_spec.children if c.type == "type_identifier")
        name = source[name_node.start_byte:name_node.end_byte].decode("utf-8")

        if doc_nodes:
            start_node = doc_nodes[0]
            doc = source[start_node.start_byte:doc_nodes[-1].end_byte].decode("utf-8")
        else:
            start_node = child
            doc = ""

        content = source[start_node.start_byte:child.end_byte].decode("utf-8")

        chunks.append(Chunk(
            id=generate_id(content),
            type=ChunkType.STRUCT,
            content=content,
            name=name,
            package_name=package_name,
            file_path=file_path,
            start_line=start_node.start_point[0] + 1,
            end_line=child.end_point[0] + 1,
            doc=doc,
            signature=f"type {name} struct",
            is_test=is_test,
            low_quality=False,
        ))

    return chunks


def _extract_methods(
    tree, source: bytes, file_path: str, package_name: str, struct_chunks: List[Chunk], is_test: bool
) -> List[Chunk]:
    """Extract method chunks and wire parent_id to their receiver struct."""
    struct_by_name = {c.name: c for c in struct_chunks}
    root = tree.root_node
    children = root.children
    chunks = []

    for i, child in enumerate(children):
        if child.type != "method_declaration":
            continue

        doc_nodes = []
        for j in range(i - 1, -1, -1):
            if children[j].type == "comment":
                doc_nodes.insert(0, children[j])
            else:
                break

        receiver_list = next(c for c in child.children if c.type == "parameter_list")
        name_node = next(c for c in child.children if c.type == "field_identifier")
        block_node = next(c for c in child.children if c.type == "block")

        name = source[name_node.start_byte:name_node.end_byte].decode("utf-8")
        receiver_type = _receiver_type_name(receiver_list, source)
        signature = source[child.start_byte:block_node.start_byte].decode("utf-8").rstrip()

        if doc_nodes:
            start_node = doc_nodes[0]
            doc = source[start_node.start_byte:doc_nodes[-1].end_byte].decode("utf-8")
        else:
            start_node = child
            doc = ""

        content = source[start_node.start_byte:child.end_byte].decode("utf-8")

        parent_struct = struct_by_name.get(receiver_type)
        parent_id = parent_struct.id if parent_struct else None

        method_chunk = Chunk(
            id=generate_id(content),
            type=ChunkType.METHOD,
            content=content,
            name=name,
            package_name=package_name,
            file_path=file_path,
            start_line=start_node.start_point[0] + 1,
            end_line=child.end_point[0] + 1,
            doc=doc,
            signature=signature,
            parent_id=parent_id,
            is_test=is_test,
            low_quality=False,
        )
        chunks.append(method_chunk)

        if parent_struct is not None:
            parent_struct.children_ids.append(method_chunk.id)

    return chunks


def _receiver_type_name(receiver_list, source: bytes) -> str:
    """Extract the base type name from a method receiver parameter list."""
    param = next((c for c in receiver_list.children if c.type == "parameter_declaration"), None)
    if param is None:
        return ""
    for child in param.children:
        if child.type == "type_identifier":
            return source[child.start_byte:child.end_byte].decode("utf-8")
        if child.type == "pointer_type":
            tid = next((c for c in child.children if c.type == "type_identifier"), None)
            if tid:
                return source[tid.start_byte:tid.end_byte].decode("utf-8")
    return ""


def _extract_interfaces(tree, source: bytes, file_path: str, package_name: str, is_test: bool) -> List[Chunk]:
    """Extract interface type declaration chunks."""
    root = tree.root_node
    children = root.children
    chunks = []

    for i, child in enumerate(children):
        if child.type != "type_declaration":
            continue
        type_spec = next((c for c in child.children if c.type == "type_spec"), None)
        if type_spec is None or not any(c.type == "interface_type" for c in type_spec.children):
            continue

        doc_nodes = []
        for j in range(i - 1, -1, -1):
            if children[j].type == "comment":
                doc_nodes.insert(0, children[j])
            else:
                break

        name_node = next(c for c in type_spec.children if c.type == "type_identifier")
        name = source[name_node.start_byte:name_node.end_byte].decode("utf-8")

        if doc_nodes:
            start_node = doc_nodes[0]
            doc = source[start_node.start_byte:doc_nodes[-1].end_byte].decode("utf-8")
        else:
            start_node = child
            doc = ""

        content = source[start_node.start_byte:child.end_byte].decode("utf-8")

        chunks.append(Chunk(
            id=generate_id(content),
            type=ChunkType.INTERFACE,
            content=content,
            name=name,
            package_name=package_name,
            file_path=file_path,
            start_line=start_node.start_point[0] + 1,
            end_line=child.end_point[0] + 1,
            doc=doc,
            signature=f"type {name} interface",
            is_test=is_test,
            low_quality=False,
        ))

    return chunks


def _extract_consts(tree, source: bytes, file_path: str, package_name: str, is_test: bool) -> List[Chunk]:
    """Extract individual const declaration chunks (not blocks)."""
    root = tree.root_node
    children = root.children
    chunks = []

    for i, child in enumerate(children):
        if child.type != "const_declaration":
            continue
        if any(c.type == "(" for c in child.children):
            continue

        doc_nodes = []
        for j in range(i - 1, -1, -1):
            if children[j].type == "comment":
                doc_nodes.insert(0, children[j])
            else:
                break

        const_spec = next(c for c in child.children if c.type == "const_spec")
        name_node = next(c for c in const_spec.children if c.type == "identifier")
        name = source[name_node.start_byte:name_node.end_byte].decode("utf-8")
        signature = source[child.start_byte:child.end_byte].decode("utf-8")

        if doc_nodes:
            start_node = doc_nodes[0]
            doc = source[start_node.start_byte:doc_nodes[-1].end_byte].decode("utf-8")
        else:
            start_node = child
            doc = ""

        content = source[start_node.start_byte:child.end_byte].decode("utf-8")

        chunks.append(Chunk(
            id=generate_id(content),
            type=ChunkType.CONST,
            content=content,
            name=name,
            package_name=package_name,
            file_path=file_path,
            start_line=start_node.start_point[0] + 1,
            end_line=child.end_point[0] + 1,
            doc=doc,
            signature=signature,
            is_test=is_test,
            low_quality=False,
        ))

    return chunks


def _extract_vars(tree, source: bytes, file_path: str, package_name: str, is_test: bool) -> List[Chunk]:
    """Extract individual var declaration chunks (not blocks)."""
    root = tree.root_node
    children = root.children
    chunks = []

    for i, child in enumerate(children):
        if child.type != "var_declaration":
            continue
        if any(c.type == "var_spec_list" for c in child.children):
            continue

        doc_nodes = []
        for j in range(i - 1, -1, -1):
            if children[j].type == "comment":
                doc_nodes.insert(0, children[j])
            else:
                break

        var_spec = next(c for c in child.children if c.type == "var_spec")
        name_node = next(c for c in var_spec.children if c.type == "identifier")
        name = source[name_node.start_byte:name_node.end_byte].decode("utf-8")
        signature = source[child.start_byte:child.end_byte].decode("utf-8")

        if doc_nodes:
            start_node = doc_nodes[0]
            doc = source[start_node.start_byte:doc_nodes[-1].end_byte].decode("utf-8")
        else:
            start_node = child
            doc = ""

        content = source[start_node.start_byte:child.end_byte].decode("utf-8")

        chunks.append(Chunk(
            id=generate_id(content),
            type=ChunkType.VAR,
            content=content,
            name=name,
            package_name=package_name,
            file_path=file_path,
            start_line=start_node.start_point[0] + 1,
            end_line=child.end_point[0] + 1,
            doc=doc,
            signature=signature,
            is_test=is_test,
            low_quality=False,
        ))

    return chunks


def _extract_blocks(tree, source: bytes, file_path: str, package_name: str, is_test: bool) -> List[Chunk]:
    """Extract grouped const or var block chunks."""
    root = tree.root_node
    children = root.children
    chunks = []

    for i, child in enumerate(children):
        is_const_block = child.type == "const_declaration" and any(
            c.type == "(" for c in child.children
        )
        is_var_block = child.type == "var_declaration" and any(
            c.type == "var_spec_list" for c in child.children
        )
        if not (is_const_block or is_var_block):
            continue

        # Collect comments above the block
        doc_nodes = []
        for j in range(i - 1, -1, -1):
            if children[j].type == "comment":
                doc_nodes.insert(0, children[j])
            else:
                break

        # Also collect inline comments within the block
        inline_comments = []
        def collect_comments(node):
            for c in node.children:
                if c.type == "comment":
                    inline_comments.append(source[c.start_byte:c.end_byte].decode("utf-8"))
                collect_comments(c)
        collect_comments(child)

        if doc_nodes:
            start_node = doc_nodes[0]
            doc = source[start_node.start_byte:doc_nodes[-1].end_byte].decode("utf-8")
        else:
            start_node = child
            doc = ""

        # Append inline comments to doc
        if inline_comments:
            inline_doc = "\n".join(inline_comments)
            doc = f"{doc}\n{inline_doc}".strip() if doc else inline_doc

        content = source[start_node.start_byte:child.end_byte].decode("utf-8")

        chunks.append(Chunk(
            id=generate_id(content),
            type=ChunkType.BLOCK,
            content=content,
            name="",
            package_name=package_name,
            file_path=file_path,
            start_line=start_node.start_point[0] + 1,
            end_line=child.end_point[0] + 1,
            doc=doc,
            signature="",
            is_test=is_test,
            low_quality=False,
        ))

    return chunks


def _extract_type_aliases(tree, source: bytes, file_path: str, package_name: str, is_test: bool) -> List[Chunk]:
    """Extract type alias and type definition chunks."""
    root = tree.root_node
    children = root.children
    chunks = []

    for i, child in enumerate(children):
        if child.type != "type_declaration":
            continue
        if not _is_type_alias(child):
            continue

        doc_nodes = []
        for j in range(i - 1, -1, -1):
            if children[j].type == "comment":
                doc_nodes.insert(0, children[j])
            else:
                break

        name = _type_alias_name(child, source)
        signature = source[child.start_byte:child.end_byte].decode("utf-8")

        if doc_nodes:
            start_node = doc_nodes[0]
            doc = source[start_node.start_byte:doc_nodes[-1].end_byte].decode("utf-8")
        else:
            start_node = child
            doc = ""

        content = source[start_node.start_byte:child.end_byte].decode("utf-8")

        chunks.append(Chunk(
            id=generate_id(content),
            type=ChunkType.TYPE_ALIAS,
            content=content,
            name=name,
            package_name=package_name,
            file_path=file_path,
            start_line=start_node.start_point[0] + 1,
            end_line=child.end_point[0] + 1,
            doc=doc,
            signature=signature,
            is_test=is_test,
            low_quality=False,
        ))

    return chunks


def _is_type_alias(node) -> bool:
    """Return True if a type_declaration is an alias or plain type definition (not struct/interface)."""
    if any(c.type == "type_alias" for c in node.children):
        return True
    type_spec = next((c for c in node.children if c.type == "type_spec"), None)
    if type_spec is not None:
        has_struct = any(c.type == "struct_type" for c in type_spec.children)
        has_iface = any(c.type == "interface_type" for c in type_spec.children)
        return not has_struct and not has_iface
    return False


def _type_alias_name(node, source: bytes) -> str:
    """Extract the declared name from a type_alias or type_spec node."""
    ta = next((c for c in node.children if c.type == "type_alias"), None)
    if ta:
        tid = next(c for c in ta.children if c.type == "type_identifier")
        return source[tid.start_byte:tid.end_byte].decode("utf-8")
    ts = next((c for c in node.children if c.type == "type_spec"), None)
    if ts:
        tid = next(c for c in ts.children if c.type == "type_identifier")
        return source[tid.start_byte:tid.end_byte].decode("utf-8")
    return ""
