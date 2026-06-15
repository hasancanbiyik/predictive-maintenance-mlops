# Outputs you'll need after `terraform apply`.
#
# Run `terraform output` (no args) to see them all, or `terraform output -raw <name>`
# for shell interpolation.

output "region" {
  description = "AWS region the stack was deployed in."
  value       = var.region
}

output "cluster_name" {
  description = "EKS cluster name -- use this with `aws eks update-kubeconfig`."
  value       = module.eks.cluster_name
}

output "cluster_endpoint" {
  description = "EKS API server URL."
  value       = module.eks.cluster_endpoint
}

output "ecr_repository_url" {
  description = "ECR repo URL for docker push / image references in k8s manifests."
  value       = aws_ecr_repository.api.repository_url
}

# Quality-of-life: print the two commands you'll always want after apply.
output "kubeconfig_command" {
  description = "Run this once to point kubectl at the new cluster."
  value       = "aws eks update-kubeconfig --region ${var.region} --name ${module.eks.cluster_name}"
}

output "ecr_login_command" {
  description = "Run this to authenticate docker against your ECR registry."
  value       = "aws ecr get-login-password --region ${var.region} | docker login --username AWS --password-stdin ${split("/", aws_ecr_repository.api.repository_url)[0]}"
}
