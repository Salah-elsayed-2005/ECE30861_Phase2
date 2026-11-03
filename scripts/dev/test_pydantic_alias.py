"""Test how Pydantic handles the regex field with alias"""
from pydantic import BaseModel, Field, ConfigDict
from typing import Optional

class ArtifactRegEx(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra='allow')
    regex: Optional[str] = Field(default=None, alias="RegEx")
    
    def get_regex_pattern(self):
        if self.regex:
            return self.regex
        extra_data = getattr(self, '__pydantic_extra__', {}) or {}
        return extra_data.get('regex', None) or extra_data.get('RegEx', None)

# Test with lowercase 'regex' (per OpenAPI spec - what autograder likely sends)
print("Test 1: lowercase 'regex' with value")
test1 = ArtifactRegEx(**{'regex': 'test-pattern'})
print(f"  regex attr = {test1.regex!r}")
print(f"  get_regex_pattern() = {test1.get_regex_pattern()!r}")

print("\nTest 2: alias 'RegEx' with value")
test2 = ArtifactRegEx(**{'RegEx': 'test-pattern'})
print(f"  regex attr = {test2.regex!r}")
print(f"  get_regex_pattern() = {test2.get_regex_pattern()!r}")

print("\nTest 3: empty string regex")
test3 = ArtifactRegEx(**{'regex': ''})
print(f"  regex attr = {test3.regex!r}")
print(f"  get_regex_pattern() = {test3.get_regex_pattern()!r}")

print("\nTest 4: null regex")
test4 = ArtifactRegEx(**{'regex': None})
print(f"  regex attr = {test4.regex!r}")
print(f"  get_regex_pattern() = {test4.get_regex_pattern()!r}")

print("\nTest 5: missing regex field (empty body)")
test5 = ArtifactRegEx(**{})
print(f"  regex attr = {test5.regex!r}")
print(f"  get_regex_pattern() = {test5.get_regex_pattern()!r}")

print("\nTest 6: unknown field only")
test6 = ArtifactRegEx(**{'other': 'value'})
print(f"  regex attr = {test6.regex!r}")
print(f"  get_regex_pattern() = {test6.get_regex_pattern()!r}")
print(f"  extra = {getattr(test6, '__pydantic_extra__', {})}")
