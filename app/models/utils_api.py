from typing import Union
from pydantic import BaseModel, Field

class GetVRAMResponse(BaseModel):
	"""Get current VRAM usage."""
	gpu_name: str = Field(..., description='Name of GPU.')
	used: float = Field(..., description='Used VRAM in GB.')
	total: float = Field(..., description='Total VRAM in GB.')
	free: float = Field(..., description='Free VRAM in GB.')
