# Deploying Vipo to AWS Free Tier

This guide provides step-by-step instructions for deploying Vipo to AWS Free Tier.

## AWS Free Tier Resources

AWS Free Tier includes:
- 750 hours/month of t2.micro EC2 instance (enough for 1 instance running 24/7)
- 30GB of EBS storage
- 5GB of S3 storage

## Step 1: Set Up AWS Account

1. Create an AWS account at [aws.amazon.com/free](https://aws.amazon.com/free/)
2. Set up billing alerts to avoid unexpected charges:
   - Go to AWS Billing Dashboard
   - Enable billing alerts
   - Create a budget with notifications

## Step 2: Launch EC2 Instance

1. Navigate to EC2 in AWS Console
2. Launch a new t2.micro instance (Free Tier eligible)
3. Select Amazon Linux 2023 AMI
4. Configure security group:
   - Allow SSH (port 22)
   - Allow HTTP (port 80) 
   - Allow custom TCP (ports 8090 and 9000)
5. Create and download key pair for SSH access
6. Launch the instance

## Step 3: Connect to EC2 Instance

```bash
ssh -i your-key.pem ec2-user@your-ec2-public-ip
```

Note: On Windows, you may need to set proper permissions on your key file:
```powershell
icacls "path\to\your-key.pem" /inheritance:r /grant:r "username:F"
```

## Step 4: Set Up Environment

```bash
# Update system
sudo yum update -y

# Install Git
sudo yum install git -y

# Install Python 3 and development tools
sudo yum install python3 python3-pip python3-devel -y

# Install additional dependencies
sudo yum install gcc -y
```

## Step 5: Deploy Vipo Application

```bash
# Clone repository (or upload your files)
git clone https://github.com/yourusername/vipo.git
cd vipo

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Set up environment variables
cp env.template .env
nano .env  # Edit with your API keys
```

## Step 6: Set Up ChromaDB

```bash
# Create directories
mkdir -p .chroma documents
```

## Step 7: Process Documents

```bash
# Run ingestion script
python ingest.py
```

## Step 8: Create Systemd Service Files

For the dashboard:
```bash
sudo tee /etc/systemd/system/vipo-dashboard.service > /dev/null << 'EOF'
[Unit]
Description=Vipo Dashboard
After=network.target

[Service]
User=ec2-user
WorkingDirectory=/home/ec2-user/vipo
Environment="PATH=/home/ec2-user/vipo/venv/bin"
ExecStart=/home/ec2-user/vipo/venv/bin/python dashboard/main.py
Restart=always

[Install]
WantedBy=multi-user.target
EOF
```

For the Chainlit app:
```bash
sudo tee /etc/systemd/system/vipo-chainlit.service > /dev/null << 'EOF'
[Unit]
Description=Vipo Chainlit App
After=network.target

[Service]
User=ec2-user
WorkingDirectory=/home/ec2-user/vipo
Environment="PATH=/home/ec2-user/vipo/venv/bin"
ExecStart=/home/ec2-user/vipo/venv/bin/chainlit run app.py --port 9000
Restart=always

[Install]
WantedBy=multi-user.target
EOF
```

## Step 9: Start and Enable Services

```bash
sudo systemctl daemon-reload
sudo systemctl start vipo-dashboard vipo-chainlit
sudo systemctl enable vipo-dashboard vipo-chainlit
```

## Step 10: Set Up Nginx as Reverse Proxy (Optional)

```bash
# Install Nginx
sudo yum install nginx -y

# Configure Nginx
sudo tee /etc/nginx/conf.d/vipo.conf > /dev/null << 'EOF'
server {
    listen 80;
    server_name _;

    location / {
        proxy_pass http://localhost:9000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
    }

    location /dashboard/ {
        proxy_pass http://localhost:8090/;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
    }
}
EOF

# Start and enable Nginx
sudo systemctl start nginx
sudo systemctl enable nginx
```

## Step 11: Access Your Application

- Chainlit app: http://your-ec2-public-ip/ or http://your-ec2-public-ip:9000
- Dashboard: http://your-ec2-public-ip/dashboard/ or http://your-ec2-public-ip:8090

## Step 12: Set Up Elastic IP (Optional)

To maintain a consistent IP address:
1. Allocate an Elastic IP in the AWS Console
2. Associate it with your EC2 instance

## Step 13: Set Up Domain Name (Optional)

1. Register a domain or use an existing one
2. Point your domain to the Elastic IP
3. Configure SSL with Let's Encrypt for secure HTTPS

## Monitoring and Maintenance

- Set up CloudWatch alarms to monitor instance health
- Create regular backups of your data
- Schedule automatic security updates

## Cost Management

- Monitor your usage to stay within Free Tier limits
- Set up billing alerts to avoid unexpected charges
- Consider using AWS Cost Explorer to track expenses

## Troubleshooting

### Application Not Starting
Check service logs:
```bash
sudo journalctl -u vipo-dashboard
sudo journalctl -u vipo-chainlit
```

### Permission Issues
Ensure the ec2-user has ownership of all files:
```bash
sudo chown -R ec2-user:ec2-user /home/ec2-user/vipo
```

### Memory Issues
If the application crashes due to memory limits on t2.micro:
```bash
# Add swap space
sudo dd if=/dev/zero of=/swapfile bs=128M count=16
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
echo '/swapfile swap swap defaults 0 0' | sudo tee -a /etc/fstab
```
