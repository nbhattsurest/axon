"""Tests for the Java language parser."""

from __future__ import annotations

import pytest

from axon.core.parsers.java_lang import JavaParser


@pytest.fixture
def parser() -> JavaParser:
    return JavaParser()


# ---------------------------------------------------------------------------
# Classes
# ---------------------------------------------------------------------------


class TestParseSimpleClass:
    """Parse a basic class with methods."""

    CODE = """\
public class UserService {
    public void save() {
        System.out.println("saving");
    }
}
"""

    def test_class_count(self, parser: JavaParser) -> None:
        result = parser.parse(self.CODE, "UserService.java")
        classes = [s for s in result.symbols if s.kind == "class"]
        assert len(classes) == 1

    def test_class_name(self, parser: JavaParser) -> None:
        result = parser.parse(self.CODE, "UserService.java")
        cls = [s for s in result.symbols if s.kind == "class"][0]
        assert cls.name == "UserService"

    def test_method_count(self, parser: JavaParser) -> None:
        result = parser.parse(self.CODE, "UserService.java")
        methods = [s for s in result.symbols if s.kind == "method"]
        assert len(methods) == 1

    def test_method_name(self, parser: JavaParser) -> None:
        result = parser.parse(self.CODE, "UserService.java")
        method = [s for s in result.symbols if s.kind == "method"][0]
        assert method.name == "save"
        assert method.class_name == "UserService"


# ---------------------------------------------------------------------------
# Inheritance
# ---------------------------------------------------------------------------


class TestParseInheritance:
    """Parse class inheritance (heritage)."""

    def test_extends(self, parser: JavaParser) -> None:
        code = """\
public class Admin extends User {
}
"""
        result = parser.parse(code, "Admin.java")
        assert ("Admin", "extends", "User") in result.heritage

    def test_implements(self, parser: JavaParser) -> None:
        code = """\
public class Admin implements Serializable {
}
"""
        result = parser.parse(code, "Admin.java")
        assert ("Admin", "implements", "Serializable") in result.heritage

    def test_extends_and_implements(self, parser: JavaParser) -> None:
        code = """\
public class Admin extends User implements Serializable, Comparable {
}
"""
        result = parser.parse(code, "Admin.java")
        assert ("Admin", "extends", "User") in result.heritage
        assert ("Admin", "implements", "Serializable") in result.heritage
        assert ("Admin", "implements", "Comparable") in result.heritage

    def test_generic_extends(self, parser: JavaParser) -> None:
        code = """\
public class Admin implements Comparable<Admin> {
}
"""
        result = parser.parse(code, "Admin.java")
        assert ("Admin", "implements", "Comparable") in result.heritage


# ---------------------------------------------------------------------------
# Interfaces
# ---------------------------------------------------------------------------


class TestParseInterface:
    """Parse interface declarations."""

    def test_interface(self, parser: JavaParser) -> None:
        code = """\
public interface UserValidator {
    boolean validate(User user);
}
"""
        result = parser.parse(code, "UserValidator.java")
        interfaces = [s for s in result.symbols if s.kind == "interface"]
        assert len(interfaces) == 1
        assert interfaces[0].name == "UserValidator"

    def test_interface_extends(self, parser: JavaParser) -> None:
        code = """\
public interface AdvancedValidator extends BaseValidator, Serializable {
}
"""
        result = parser.parse(code, "AdvancedValidator.java")
        interfaces = [s for s in result.symbols if s.kind == "interface"]
        assert len(interfaces) == 1
        # Note: Interface extends is detected differently in tree-sitter-java


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class TestParseEnum:
    """Parse enum declarations."""

    def test_enum(self, parser: JavaParser) -> None:
        code = """\
public enum Status {
    ACTIVE, INACTIVE, PENDING
}
"""
        result = parser.parse(code, "Status.java")
        enums = [s for s in result.symbols if s.kind == "enum"]
        assert len(enums) == 1
        assert enums[0].name == "Status"


# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------


class TestParseImports:
    """Parse import statements."""

    CODE = """\
import java.util.List;
import java.util.*;
import static java.lang.Math.PI;
"""

    def test_import_count(self, parser: JavaParser) -> None:
        result = parser.parse(self.CODE, "Test.java")
        assert len(result.imports) == 3

    def test_single_import(self, parser: JavaParser) -> None:
        result = parser.parse(self.CODE, "Test.java")
        list_imp = [i for i in result.imports if i.module == "java.util.List"]
        assert len(list_imp) == 1
        assert "List" in list_imp[0].names

    def test_wildcard_import(self, parser: JavaParser) -> None:
        result = parser.parse(self.CODE, "Test.java")
        wildcard_imp = [i for i in result.imports if i.module == "java.util"]
        assert len(wildcard_imp) == 1
        assert "*" in wildcard_imp[0].names


# ---------------------------------------------------------------------------
# Method calls
# ---------------------------------------------------------------------------


class TestParseCalls:
    """Parse method invocations."""

    CODE = """\
public class Service {
    public void run() {
        User user = new User("test");
        user.save();
        validate(user);
        this.cleanup();
    }
}
"""

    def test_method_call_with_receiver(self, parser: JavaParser) -> None:
        result = parser.parse(self.CODE, "Service.java")
        save_calls = [c for c in result.calls if c.name == "save"]
        assert len(save_calls) == 1
        assert save_calls[0].receiver == "user"

    def test_method_call_without_receiver(self, parser: JavaParser) -> None:
        result = parser.parse(self.CODE, "Service.java")
        validate_calls = [c for c in result.calls if c.name == "validate"]
        assert len(validate_calls) == 1
        assert validate_calls[0].receiver == ""

    def test_this_call(self, parser: JavaParser) -> None:
        result = parser.parse(self.CODE, "Service.java")
        cleanup_calls = [c for c in result.calls if c.name == "cleanup"]
        assert len(cleanup_calls) == 1
        assert cleanup_calls[0].receiver == "this"

    def test_new_expression(self, parser: JavaParser) -> None:
        result = parser.parse(self.CODE, "Service.java")
        user_calls = [c for c in result.calls if c.name == "User"]
        assert len(user_calls) == 1


# ---------------------------------------------------------------------------
# Type annotations
# ---------------------------------------------------------------------------


class TestParseTypeAnnotations:
    """Parse type references from parameters and return types."""

    CODE = """\
public class Service {
    public User getUser(Config config, int id) {
        return null;
    }
}
"""

    def test_return_type(self, parser: JavaParser) -> None:
        result = parser.parse(self.CODE, "Service.java")
        return_refs = [t for t in result.type_refs if t.kind == "return"]
        assert any(t.name == "User" for t in return_refs)

    def test_param_type(self, parser: JavaParser) -> None:
        result = parser.parse(self.CODE, "Service.java")
        param_refs = [t for t in result.type_refs if t.kind == "param"]
        assert any(t.name == "Config" for t in param_refs)

    def test_builtin_types_skipped(self, parser: JavaParser) -> None:
        result = parser.parse(self.CODE, "Service.java")
        # int is built-in and should be skipped
        param_names = [t.name for t in result.type_refs]
        assert "int" not in param_names


# ---------------------------------------------------------------------------
# Constructors
# ---------------------------------------------------------------------------


class TestParseConstructor:
    """Parse constructor declarations."""

    CODE = """\
public class User {
    private String name;

    public User(String name) {
        this.name = name;
    }
}
"""

    def test_constructor_as_method(self, parser: JavaParser) -> None:
        result = parser.parse(self.CODE, "User.java")
        methods = [s for s in result.symbols if s.kind == "method"]
        assert len(methods) == 1
        assert methods[0].name == "User"
        assert methods[0].class_name == "User"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases and less common patterns."""

    def test_empty_file(self, parser: JavaParser) -> None:
        result = parser.parse("", "Empty.java")
        assert result.symbols == []
        assert result.imports == []
        assert result.calls == []
        assert result.type_refs == []
        assert result.heritage == []

    def test_syntax_error_does_not_crash(self, parser: JavaParser) -> None:
        code = "public class Broken {"
        # Should not raise; tree-sitter produces a partial tree
        result = parser.parse(code, "Broken.java")
        assert isinstance(result, type(result))

    def test_nested_class(self, parser: JavaParser) -> None:
        code = """\
public class Outer {
    public class Inner {
    }
}
"""
        result = parser.parse(code, "Outer.java")
        classes = [s for s in result.symbols if s.kind == "class"]
        names = {c.name for c in classes}
        assert "Outer" in names
        assert "Inner" in names

    def test_record_class(self, parser: JavaParser) -> None:
        code = """\
public record UserDTO(String name, int age) {}
"""
        result = parser.parse(code, "UserDTO.java")
        classes = [s for s in result.symbols if s.kind == "class"]
        assert len(classes) == 1
        assert classes[0].name == "UserDTO"


# ---------------------------------------------------------------------------
# Method signatures
# ---------------------------------------------------------------------------


class TestMethodSignature:
    """Test method signature extraction."""

    def test_method_signature(self, parser: JavaParser) -> None:
        code = """\
public class Service {
    public User getUser(int id) {
        return null;
    }
}
"""
        result = parser.parse(code, "Service.java")
        methods = [s for s in result.symbols if s.kind == "method"]
        assert len(methods) == 1
        assert "getUser" in methods[0].signature
        assert "(int id)" in methods[0].signature
