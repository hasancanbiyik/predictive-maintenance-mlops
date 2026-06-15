# Variables for the EKS + ECR demo stack.
#
# Defaults are tuned for a *short* portfolio demo: cheap nodes, single NAT,
# small disk. They are NOT a production blueprint -- see comments below.

variable "region" {
  description = "AWS region. us-east-1 is usually cheapest."
  type        = string
  default     = "us-east-1"
}

variable "project" {
  description = "Short project name used as a prefix on every resource."
  type        = string
  default     = "pdm"
}

variable "owner" {
  description = "Free-text owner tag for cost attribution. Use your name/email."
  type        = string
  default     = "hasancanbiyik"
}

variable "cluster_version" {
  description = "EKS Kubernetes version. Bump every ~6 months."
  type        = string
  default     = "1.31"
}

variable "node_instance_type" {
  description = <<-EOT
    Worker node EC2 type. t3.medium (2 vCPU / 4 GB) is the cheapest practical
    pick for this stack -- MLflow alone wants ~512 MB. Don't go smaller.
  EOT
  type    = string
  default = "t3.medium"
}

variable "node_desired_size" {
  description = "Worker node count. 2 is enough for API replicas + MLflow + headroom."
  type        = number
  default     = 2
}

variable "vpc_cidr" {
  description = "CIDR for the demo VPC."
  type        = string
  default     = "10.0.0.0/16"
}

# Cost-saving switch. Production EKS uses ONE NAT per AZ for fault tolerance.
# For a portfolio demo that gets destroyed in an hour, one NAT in one AZ is
# fine and saves ~$32/mo if you forget to destroy.
variable "single_nat_gateway" {
  description = "Use one NAT gateway across all AZs (cheaper, less HA)."
  type        = bool
  default     = true
}
