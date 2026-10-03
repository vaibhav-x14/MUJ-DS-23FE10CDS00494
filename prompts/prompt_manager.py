import re
from pathlib import Path
from typing import Dict, Any, Optional
import yaml
from pydantic import BaseModel, Field

PROMPTS_DIR = Path(__file__).resolve().parent


class PromptTemplate(BaseModel):
    version: str
    name: str
    description: str
    system_prompt: str
    user_template: str
    output_format: str = "TEXT"
    schema_def: Optional[str] = Field(default=None, alias="schema")
    few_shot_examples: list = Field(default_factory=list)

    def render_system(self, **kwargs) -> str:
        """Render system prompt with optional parameter replacement."""
        text = self.system_prompt
        for k, v in kwargs.items():
            text = text.replace(f"{{{{{k}}}}}", str(v))
        return text

    def render_user(self, **kwargs) -> str:
        """Render user prompt template with keyword arguments."""
        text = self.user_template
        for k, v in kwargs.items():
            text = text.replace(f"{{{{{k}}}}}", str(v))
        # Warn or clean un-substituted variables
        remaining = re.findall(r"\{\{([a-zA-Z0-9_]+)\}\}", text)
        if remaining:
            for rem in remaining:
                text = text.replace(f"{{{{{rem}}}}}", "")
        return text.strip()


class PromptManager:
    """Production catalog and loader for versioned system prompts."""

    _instance = None
    _templates: Dict[str, PromptTemplate] = {}

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls)
            cls._instance._load_all_templates()
        return cls._instance

    def _load_all_templates(self):
        """Scans prompts directory and parses all YAML prompt definitions."""
        self._templates = {}
        for yaml_file in PROMPTS_DIR.glob("*.yaml"):
            try:
                with open(yaml_file, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f)
                    if isinstance(data, dict) and "name" in data and "system_prompt" in data:
                        template = PromptTemplate(**data)
                        self._templates[template.name] = template
                        # Also register by filename stem for convenience
                        self._templates[yaml_file.stem] = template
            except Exception as e:
                print(f"[Warning] Failed to load prompt file {yaml_file}: {e}")

    def get_template(self, name_or_stem: str) -> PromptTemplate:
        """Retrieve prompt template by name or filename."""
        if name_or_stem not in self._templates:
            # Try reloading once in case a new prompt was added
            self._load_all_templates()
        if name_or_stem not in self._templates:
            raise KeyError(f"Prompt template '{name_or_stem}' not found in {PROMPTS_DIR}")
        return self._templates[name_or_stem]

    def render(self, name_or_stem: str, user_vars: Dict[str, Any], system_vars: Optional[Dict[str, Any]] = None) -> tuple[str, str]:
        """Returns tuple of (system_prompt, user_prompt)."""
        tpl = self.get_template(name_or_stem)
        system_p = tpl.render_system(**(system_vars or {}))
        user_p = tpl.render_user(**user_vars)
        return system_p, user_p

    def list_templates(self) -> Dict[str, Dict[str, str]]:
        """List metadata of all available prompt templates."""
        return {
            name: {
                "version": tpl.version,
                "description": tpl.description,
                "output_format": tpl.output_format,
            }
            for name, tpl in self._templates.items()
        }


# Global helper instance
prompt_catalog = PromptManager()
