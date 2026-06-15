# Terraform: EKS + ECR for the Predictive Maintenance demo

This directory provisions the AWS infrastructure needed to run the
production-loop demo on managed Kubernetes:

- VPC with public + private subnets across 3 AZs
- EKS cluster (1.31, 2× t3.medium worker nodes)
- ECR repo for the API image
- IAM + OIDC wired by the upstream EKS module

## Cost estimate

Provisioned 24/7 this would run ~$170/mo. **You're not going to do that.**
The intended workflow is *apply → demo → destroy* in under an hour:

| Resource | Hourly | 1 hour | 24 hours |
|---|---|---|---|
| EKS control plane | $0.10 | $0.10 | $2.40 |
| 2× t3.medium nodes | $0.083 | $0.083 | $2.00 |
| NAT gateway (single) | $0.045 | $0.045 | $1.08 |
| Load balancer | $0.025 | $0.025 | $0.60 |
| **Total** | **~$0.25** | **~$0.25** | **~$6** |

ECR storage and data transfer add a few cents.

## Apply / destroy

```bash
# One-time AWS credential check
aws sts get-caller-identity

# From this terraform/ directory
terraform init
terraform plan -out=plan.out
terraform apply plan.out
# (Cluster comes up in ~15-20 minutes. EKS is slow; this is normal.)

# When done, ALWAYS run:
terraform destroy
```

## After apply: connect kubectl + push image

The exact commands are printed by `terraform output`. Typical flow:

```bash
# 1. Point kubectl at the new cluster
$(terraform output -raw kubeconfig_command)
kubectl get nodes      # should list 2 Ready nodes

# 2. Authenticate docker against ECR
$(terraform output -raw ecr_login_command)

# 3. Push the API image (built earlier from compose)
ECR_URL=$(terraform output -raw ecr_repository_url)
docker tag predictive-maintenance-api:0.7.1 ${ECR_URL}:0.7.1
docker tag predictive-maintenance-api:0.7.1 ${ECR_URL}:latest
docker push ${ECR_URL}:0.7.1
docker push ${ECR_URL}:latest

# 4. Edit k8s/api.yaml to use ECR_URL instead of the local tag, then:
sed -i.bak "s|predictive-maintenance-api:0.7.1|${ECR_URL}:0.7.1|" ../k8s/api.yaml
kubectl apply -f ../k8s/

# 5. Wait for pods, port-forward, train, hit /predict (same as kind workflow)
kubectl get pods -n pdm -w

# 6. Take screenshots / record a demo
```

## Tear-down checklist

EKS leaves *nothing* lingering when destroyed via Terraform, **but**:

- [ ] Run `terraform destroy` and watch it complete (no orange "still going" lines).
- [ ] Verify in the AWS console: EC2 → Instances (empty), VPC (your demo VPC gone),
      EKS → Clusters (empty), ECR → Repos (empty).
- [ ] Check **AWS Cost Explorer** the next day. If anything still costs money,
      it's the NAT gateway or an LB the K8s services didn't clean up before
      Terraform tried to remove the VPC.
- [ ] If a destroy hangs on a security group: usually a leftover ELB.
      Delete it via the AWS Console and retry.

## What's intentionally NOT here

- **EFS for shared PVCs.** The K8s CronJob in `k8s/cronjob.yaml` uses an
  emptyDir for the prediction log. Wiring a shared EFS volume between API
  and CronJob is doable but adds two more Terraform modules and ~$0.30/GB/mo.
  Skipped for the demo; documented as a future improvement.
- **Hosted MLflow.** MLflow runs as a pod inside the cluster (same as kind).
  Production would put it on an RDS-backed deployment with S3 artifact store.
- **Route53 + cert-manager.** No DNS / TLS. Port-forward is fine for the demo;
  Ingress + TLS would be a Phase 10.
