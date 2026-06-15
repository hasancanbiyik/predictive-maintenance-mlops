# Provider configuration.
#
# We pin Terraform >= 1.6 (modern stack) and pin AWS provider major version.
# Pinning avoids "the cluster I tore down yesterday doesn't plan-match today's
# version" -- a real issue if you destroy/recreate across weeks.

terraform {
  required_version = ">= 1.6"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.70"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.32"
    }
  }
}

provider "aws" {
  region = var.region

  default_tags {
    tags = {
      Project   = var.project
      ManagedBy = "Terraform"
      Owner     = var.owner
    }
  }
}

# The kubernetes provider talks to the cluster Terraform just provisioned.
# Its auth uses the AWS CLI's `aws eks get-token` so we don't manage long-lived
# kubeconfigs in Terraform state.
provider "kubernetes" {
  host                   = module.eks.cluster_endpoint
  cluster_ca_certificate = base64decode(module.eks.cluster_certificate_authority_data)

  exec {
    api_version = "client.authentication.k8s.io/v1beta1"
    command     = "aws"
    args        = ["eks", "get-token", "--cluster-name", module.eks.cluster_name, "--region", var.region]
  }
}
