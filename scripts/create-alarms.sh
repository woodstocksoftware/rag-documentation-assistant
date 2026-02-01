#!/bin/bash

# Configuration
TOPIC_ARN="arn:aws:sns:us-east-1:695230565273:rag-alerts-dev"
REGION="us-east-1"

echo "Creating CloudWatch Alarms..."

# 1. Query Lambda Errors
aws cloudwatch put-metric-alarm \
  --alarm-name "RAG-Query-Errors" \
  --alarm-description "Alert when Query Lambda has errors" \
  --namespace "AWS/Lambda" \
  --metric-name "Errors" \
  --dimensions Name=FunctionName,Value=rag-query-dev \
  --statistic Sum \
  --period 300 \
  --threshold 1 \
  --comparison-operator GreaterThanOrEqualToThreshold \
  --evaluation-periods 1 \
  --alarm-actions $TOPIC_ARN \
  --region $REGION

echo "✓ Created: RAG-Query-Errors"

# 2. Ingest Lambda Errors
aws cloudwatch put-metric-alarm \
  --alarm-name "RAG-Ingest-Errors" \
  --alarm-description "Alert when Ingest Lambda has errors" \
  --namespace "AWS/Lambda" \
  --metric-name "Errors" \
  --dimensions Name=FunctionName,Value=rag-ingest-dev \
  --statistic Sum \
  --period 300 \
  --threshold 1 \
  --comparison-operator GreaterThanOrEqualToThreshold \
  --evaluation-periods 1 \
  --alarm-actions $TOPIC_ARN \
  --region $REGION

echo "✓ Created: RAG-Ingest-Errors"

# 3. Query Lambda High Latency (>10 seconds average)
aws cloudwatch put-metric-alarm \
  --alarm-name "RAG-Query-HighLatency" \
  --alarm-description "Alert when Query Lambda is slow (>10s average)" \
  --namespace "AWS/Lambda" \
  --metric-name "Duration" \
  --dimensions Name=FunctionName,Value=rag-query-dev \
  --statistic Average \
  --period 300 \
  --threshold 10000 \
  --comparison-operator GreaterThanThreshold \
  --evaluation-periods 3 \
  --alarm-actions $TOPIC_ARN \
  --region $REGION

echo "✓ Created: RAG-Query-HighLatency"

# 4. API Gateway 5xx Errors
aws cloudwatch put-metric-alarm \
  --alarm-name "RAG-API-5xxErrors" \
  --alarm-description "Alert when API returns 5xx errors" \
  --namespace "AWS/ApiGateway" \
  --metric-name "5XXError" \
  --dimensions Name=ApiName,Value=rag-api-dev \
  --statistic Sum \
  --period 300 \
  --threshold 1 \
  --comparison-operator GreaterThanOrEqualToThreshold \
  --evaluation-periods 1 \
  --alarm-actions $TOPIC_ARN \
  --region $REGION

echo "✓ Created: RAG-API-5xxErrors"

# 5. API Gateway 4xx Errors (client errors, including auth failures)
aws cloudwatch put-metric-alarm \
  --alarm-name "RAG-API-4xxErrors" \
  --alarm-description "Alert on high rate of client errors (auth failures, bad requests)" \
  --namespace "AWS/ApiGateway" \
  --metric-name "4XXError" \
  --dimensions Name=ApiName,Value=rag-api-dev \
  --statistic Sum \
  --period 300 \
  --threshold 50 \
  --comparison-operator GreaterThanThreshold \
  --evaluation-periods 1 \
  --alarm-actions $TOPIC_ARN \
  --region $REGION

echo "✓ Created: RAG-API-4xxErrors"

# 6. SQS Queue Backlog
aws cloudwatch put-metric-alarm \
  --alarm-name "RAG-SQS-Backlog" \
  --alarm-description "Alert when documents are stuck in processing queue" \
  --namespace "AWS/SQS" \
  --metric-name "ApproximateNumberOfMessagesVisible" \
  --dimensions Name=QueueName,Value=rag-document-processing-dev \
  --statistic Average \
  --period 300 \
  --threshold 10 \
  --comparison-operator GreaterThanThreshold \
  --evaluation-periods 3 \
  --alarm-actions $TOPIC_ARN \
  --region $REGION

echo "✓ Created: RAG-SQS-Backlog"

# 7. Lambda Throttling
aws cloudwatch put-metric-alarm \
  --alarm-name "RAG-Lambda-Throttled" \
  --alarm-description "Alert when Lambda functions are being throttled" \
  --namespace "AWS/Lambda" \
  --metric-name "Throttles" \
  --dimensions Name=FunctionName,Value=rag-query-dev \
  --statistic Sum \
  --period 300 \
  --threshold 1 \
  --comparison-operator GreaterThanOrEqualToThreshold \
  --evaluation-periods 1 \
  --alarm-actions $TOPIC_ARN \
  --region $REGION

echo "✓ Created: RAG-Lambda-Throttled"

echo ""
echo "✅ All alarms created successfully!"
echo ""
echo "View alarms: https://us-east-1.console.aws.amazon.com/cloudwatch/home?region=us-east-1#alarmsV2:"