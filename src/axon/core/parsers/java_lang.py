"""Java language parser using tree-sitter.

Extracts symbols (classes, interfaces, enums, methods), imports, call
expressions, type annotation references, and heritage (extends/implements)
relationships from Java source files.
"""

from __future__ import annotations

import tree_sitter_java as tsjava
from tree_sitter import Language, Node, Parser

from axon.core.parsers.base import (
    CallInfo,
    ImportInfo,
    LanguageParser,
    ParseResult,
    SymbolInfo,
    TypeRef,
)

JAVA_LANGUAGE = Language(tsjava.language())

_BUILTIN_TYPES: frozenset[str] = frozenset(
    {
        "int",
        "long",
        "short",
        "byte",
        "float",
        "double",
        "boolean",
        "char",
        "void",
        "String",
        "Object",
        "Integer",
        "Long",
        "Short",
        "Byte",
        "Float",
        "Double",
        "Boolean",
        "Character",
        "Void",
    }
)


class JavaParser(LanguageParser):
    """Parse Java source files via tree-sitter."""

    def __init__(self) -> None:
        self._parser = Parser(JAVA_LANGUAGE)

    def parse(self, content: str, file_path: str) -> ParseResult:
        """Parse *content* and return an intermediate :class:`ParseResult`."""
        tree = self._parser.parse(content.encode("utf-8"))

        result = ParseResult()
        self._walk(tree.root_node, content, result)
        return result

    def _walk(
        self, node: Node, source: str, result: ParseResult, visited: set[int] | None = None
    ) -> None:
        """Walk the tree recursively, dispatching on node type."""
        if visited is None:
            visited = set()

        node_key = node.id
        if node_key in visited:
            return
        visited.add(node_key)

        ntype = node.type

        if ntype == "class_declaration":
            self._extract_class(node, source, result)
        elif ntype == "interface_declaration":
            self._extract_interface(node, source, result)
        elif ntype == "enum_declaration":
            self._extract_enum(node, source, result)
        elif ntype == "record_declaration":
            self._extract_record(node, source, result)
        elif ntype == "method_declaration":
            self._extract_method(node, source, result)
        elif ntype == "constructor_declaration":
            self._extract_constructor(node, source, result)
        elif ntype == "import_declaration":
            self._extract_import(node, source, result)
        elif ntype == "method_invocation":
            self._extract_call(node, source, result)
        elif ntype == "object_creation_expression":
            self._extract_new_expression(node, source, result)

        for child in node.children:
            self._walk(child, source, result, visited)

    def _extract_class(self, node: Node, source: str, result: ParseResult) -> None:
        """Extract a class declaration."""
        name_node = node.child_by_field_name("name")
        if name_node is None:
            return

        name = name_node.text.decode()
        start_line = node.start_point[0] + 1
        end_line = node.end_point[0] + 1
        content = node.text.decode()

        result.symbols.append(
            SymbolInfo(
                name=name,
                kind="class",
                start_line=start_line,
                end_line=end_line,
                content=content,
            )
        )

        # Extract heritage (extends/implements)
        for child in node.children:
            if child.type == "superclass":
                self._extract_superclass(name, child, result)
            elif child.type == "super_interfaces":
                self._extract_super_interfaces(name, child, result)

    def _extract_superclass(
        self, class_name: str, superclass_node: Node, result: ParseResult
    ) -> None:
        """Extract 'extends' relationship from superclass node."""
        for child in superclass_node.children:
            if child.type == "type_identifier":
                result.heritage.append((class_name, "extends", child.text.decode()))
            elif child.type == "generic_type":
                type_id = self._extract_type_identifier(child)
                if type_id:
                    result.heritage.append((class_name, "extends", type_id))

    def _extract_super_interfaces(
        self, class_name: str, interfaces_node: Node, result: ParseResult
    ) -> None:
        """Extract 'implements' relationships from super_interfaces node."""
        for child in interfaces_node.children:
            if child.type == "type_list":
                for type_child in child.children:
                    if type_child.type == "type_identifier":
                        result.heritage.append(
                            (class_name, "implements", type_child.text.decode())
                        )
                    elif type_child.type == "generic_type":
                        type_id = self._extract_type_identifier(type_child)
                        if type_id:
                            result.heritage.append((class_name, "implements", type_id))

    def _extract_interface(self, node: Node, source: str, result: ParseResult) -> None:
        """Extract an interface declaration."""
        name_node = node.child_by_field_name("name")
        if name_node is None:
            return

        name = name_node.text.decode()
        start_line = node.start_point[0] + 1
        end_line = node.end_point[0] + 1
        content = node.text.decode()

        result.symbols.append(
            SymbolInfo(
                name=name,
                kind="interface",
                start_line=start_line,
                end_line=end_line,
                content=content,
            )
        )

        # Extract 'extends' relationships for interface
        for child in node.children:
            if child.type == "extends_interfaces":
                for sub in child.children:
                    if sub.type == "type_list":
                        for type_child in sub.children:
                            if type_child.type == "type_identifier":
                                result.heritage.append(
                                    (name, "extends", type_child.text.decode())
                                )
                            elif type_child.type == "generic_type":
                                type_id = self._extract_type_identifier(type_child)
                                if type_id:
                                    result.heritage.append((name, "extends", type_id))

    def _extract_enum(self, node: Node, source: str, result: ParseResult) -> None:
        """Extract an enum declaration."""
        name_node = node.child_by_field_name("name")
        if name_node is None:
            return

        name = name_node.text.decode()
        start_line = node.start_point[0] + 1
        end_line = node.end_point[0] + 1
        content = node.text.decode()

        result.symbols.append(
            SymbolInfo(
                name=name,
                kind="enum",
                start_line=start_line,
                end_line=end_line,
                content=content,
            )
        )

    def _extract_record(self, node: Node, source: str, result: ParseResult) -> None:
        """Extract a record declaration (Java 14+)."""
        name_node = node.child_by_field_name("name")
        if name_node is None:
            return

        name = name_node.text.decode()
        start_line = node.start_point[0] + 1
        end_line = node.end_point[0] + 1
        content = node.text.decode()

        result.symbols.append(
            SymbolInfo(
                name=name,
                kind="class",
                start_line=start_line,
                end_line=end_line,
                content=content,
            )
        )

    def _extract_method(self, node: Node, source: str, result: ParseResult) -> None:
        """Extract a method declaration."""
        name_node = node.child_by_field_name("name")
        if name_node is None:
            return

        name = name_node.text.decode()
        start_line = node.start_point[0] + 1
        end_line = node.end_point[0] + 1
        content = node.text.decode()

        class_name = self._find_parent_class_name(node)
        signature = self._build_method_signature(node, name)

        result.symbols.append(
            SymbolInfo(
                name=name,
                kind="method",
                start_line=start_line,
                end_line=end_line,
                content=content,
                signature=signature,
                class_name=class_name,
            )
        )

        # Extract return type reference
        type_node = node.child_by_field_name("type")
        if type_node is not None:
            type_name = self._get_type_name(type_node)
            if type_name and type_name not in _BUILTIN_TYPES:
                result.type_refs.append(
                    TypeRef(
                        name=type_name,
                        kind="return",
                        line=type_node.start_point[0] + 1,
                    )
                )

        # Extract parameter type references
        params_node = node.child_by_field_name("parameters")
        if params_node is not None:
            self._extract_parameter_types(params_node, result)

    def _extract_constructor(self, node: Node, source: str, result: ParseResult) -> None:
        """Extract a constructor declaration."""
        name_node = node.child_by_field_name("name")
        if name_node is None:
            return

        name = name_node.text.decode()
        start_line = node.start_point[0] + 1
        end_line = node.end_point[0] + 1
        content = node.text.decode()

        class_name = self._find_parent_class_name(node)
        signature = self._build_constructor_signature(node, name)

        result.symbols.append(
            SymbolInfo(
                name=name,
                kind="method",
                start_line=start_line,
                end_line=end_line,
                content=content,
                signature=signature,
                class_name=class_name,
            )
        )

        # Extract parameter type references
        params_node = node.child_by_field_name("parameters")
        if params_node is not None:
            self._extract_parameter_types(params_node, result)

    def _extract_parameter_types(
        self, params_node: Node, result: ParseResult
    ) -> None:
        """Extract type references from method/constructor parameters."""
        for child in params_node.children:
            if child.type == "formal_parameter":
                type_node = child.child_by_field_name("type")
                name_node = child.child_by_field_name("name")
                if type_node is not None:
                    type_name = self._get_type_name(type_node)
                    param_name = name_node.text.decode() if name_node else ""
                    if type_name and type_name not in _BUILTIN_TYPES:
                        result.type_refs.append(
                            TypeRef(
                                name=type_name,
                                kind="param",
                                line=type_node.start_point[0] + 1,
                                param_name=param_name,
                            )
                        )

    def _extract_import(self, node: Node, source: str, result: ParseResult) -> None:
        """Extract an import declaration."""
        is_wildcard = False
        module = ""

        for child in node.children:
            if child.type == "scoped_identifier":
                module = child.text.decode()
            elif child.type == "asterisk":
                is_wildcard = True

        if not module:
            return

        # Extract the last component as the imported name
        parts = module.rsplit(".", 1)
        if is_wildcard:
            names = ["*"]
        else:
            names = [parts[-1]] if len(parts) > 1 else [module]

        result.imports.append(
            ImportInfo(
                module=module,
                names=names,
                is_relative=False,  # Java doesn't have relative imports
            )
        )

    def _extract_call(self, node: Node, source: str, result: ParseResult) -> None:
        """Extract a method invocation."""
        name_node = node.child_by_field_name("name")
        if name_node is None:
            return

        name = name_node.text.decode()
        line = node.start_point[0] + 1
        arguments = self._extract_identifier_arguments(node)

        object_node = node.child_by_field_name("object")
        receiver = ""
        if object_node is not None:
            if object_node.type == "this":
                receiver = "this"
            elif object_node.type == "identifier":
                receiver = object_node.text.decode()
            elif object_node.type == "method_invocation":
                # Chained call: obj.method1().method2()
                receiver = object_node.text.decode()

        result.calls.append(
            CallInfo(
                name=name,
                line=line,
                receiver=receiver,
                arguments=arguments,
            )
        )

    def _extract_new_expression(
        self, node: Node, source: str, result: ParseResult
    ) -> None:
        """Extract an object creation expression (new ClassName(...))."""
        type_node = node.child_by_field_name("type")
        if type_node is None:
            return

        type_name = self._get_type_name(type_node)
        if not type_name:
            return

        line = node.start_point[0] + 1
        arguments = self._extract_identifier_arguments(node)

        result.calls.append(
            CallInfo(
                name=type_name,
                line=line,
                arguments=arguments,
            )
        )

    @staticmethod
    def _extract_identifier_arguments(call_node: Node) -> list[str]:
        """Extract bare identifier arguments from a call or object creation."""
        args_node = call_node.child_by_field_name("arguments")
        if args_node is None:
            return []

        identifiers: list[str] = []
        for child in args_node.children:
            if child.type == "identifier":
                identifiers.append(child.text.decode())
        return identifiers

    @staticmethod
    def _get_type_name(type_node: Node) -> str:
        """Extract the simple type name from a type node."""
        if type_node.type == "type_identifier":
            return type_node.text.decode()
        if type_node.type == "generic_type":
            for child in type_node.children:
                if child.type == "type_identifier":
                    return child.text.decode()
        if type_node.type in ("integral_type", "floating_point_type", "boolean_type"):
            return type_node.text.decode()
        if type_node.type == "void_type":
            return "void"
        if type_node.type == "array_type":
            # For arrays like String[], get the element type
            for child in type_node.children:
                if child.type == "type_identifier":
                    return child.text.decode()
        return ""

    @staticmethod
    def _extract_type_identifier(generic_node: Node) -> str:
        """Extract the type identifier from a generic_type node."""
        for child in generic_node.children:
            if child.type == "type_identifier":
                return child.text.decode()
        return ""

    @staticmethod
    def _build_method_signature(node: Node, name: str) -> str:
        """Build a human-readable signature for a method."""
        params_node = node.child_by_field_name("parameters")
        type_node = node.child_by_field_name("type")

        params_text = params_node.text.decode() if params_node else "()"
        return_type = ""
        if type_node is not None:
            return_type = type_node.text.decode()

        return f"{return_type} {name}{params_text}".strip()

    @staticmethod
    def _build_constructor_signature(node: Node, name: str) -> str:
        """Build a human-readable signature for a constructor."""
        params_node = node.child_by_field_name("parameters")
        params_text = params_node.text.decode() if params_node else "()"
        return f"{name}{params_text}"

    @staticmethod
    def _find_parent_class_name(node: Node) -> str:
        """Walk up the tree to find the enclosing class name."""
        current = node.parent
        while current is not None:
            if current.type in ("class_declaration", "interface_declaration",
                                "enum_declaration", "record_declaration"):
                name_node = current.child_by_field_name("name")
                if name_node is not None:
                    return name_node.text.decode()
            current = current.parent
        return ""
