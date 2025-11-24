variable "project" {
  type    = string
  default = "tmr"
}

variable "env" {
  type    = string
  default = "dev"
}

variable "region" {
  type    = string
  default = "us-east-1"
}

variable "hf_token" {
  type        = string
  description = "HuggingFace API token"
  default     = ""
  sensitive   = true
}

variable "gen_ai_studio_api_key" {
  type        = string
  description = "Purdue GenAI Studio API key"
  default     = ""
  sensitive   = true
}
