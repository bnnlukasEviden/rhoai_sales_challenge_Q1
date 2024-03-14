
Helm-Chart Configurations:
https://github.com/strangiato/helm-charts/tree/main/charts/mlflow-server#values

Helm-Chart Installation in RHOAI Namespace (redhat-ods-applications):
helm upgrade -i mlflow-server strangiato/mlflow-server --set odhApplication.enabled=true --set objectStorage.mlflowBucketName=sc0124-mlflow --set objectStorage.s3EndpointUrl=https://s3-eu-central-2.ionoscloud.com --set objectStorage.s3AccessKeyId='' --set objectStorage.s3SecretAccessKey='' --set objectStorage.objectBucketClaim.enabled=false

