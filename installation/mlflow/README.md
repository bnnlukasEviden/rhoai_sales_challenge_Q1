
#### Helm-Chart Configurations:

https://github.com/strangiato/helm-charts/tree/main/charts/mlflow-server#values

#### Helm-Chart Installation in RHOAI Namespace (redhat-ods-applications):

Requirement: CrunchyDB Operator
~~~
helm upgrade -i mlflow-server strangiato/mlflow-server --set odhApplication.enabled=true --set objectStorage.mlflowBucketName=sc0124-mlflow --set objectStorage.s3EndpointUrl=https://s3-eu-central-2.ionoscloud.com --set objectStorage.s3AccessKeyId='' --set objectStorage.s3SecretAccessKey='' --set objectStorage.objectBucketClaim.enabled=false --set crunchyPostgres.enabled=false
~~~

! Installed PostgresCluster isn't working out-of-the-box, use the postgresCluster.yaml instead

! After change config in MLFlow Deployment accordingly (database env variables)