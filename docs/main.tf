terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
  required_version = ">= 1.2"
}

provider "aws" {
  region = "us-west-2"   # Change if needed; us-east-1 or us-west-2 often have good Landsat proximity
}

data "aws_ami" "al2023" {
  most_recent = true

  filter {
    name   = "name"
    values = ["al2023-ami-2023.*-arm64"]
  }

  owners = ["amazon"] # Canonical
}

# Security Group - SSH + outbound internet
resource "aws_security_group" "cbi_sg" {
  name        = "cbi-compute-sg-west"
  description = "Security group for CBI computation EC2"

  ingress {
    description = "SSH from your IP"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    description = "Allow all outbound (for Landsat downloads)"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name        = "cbi-compute-sg"
    Environment = "dev"
  }
}

# EC2 Instance
resource "aws_instance" "cbi_compute" {
  ami           = data.aws_ami.al2023.id
  instance_type = "c7g.4xlarge"
  key_name = "cbi_server_key"

  root_block_device {
    volume_size           = 30    # Free tier max
    volume_type           = "gp3"
    delete_on_termination = true
  }

  vpc_security_group_ids = [aws_security_group.cbi_sg.id]
  associate_public_ip_address = true

  # SETUP FOR DEVELOPEMENT
  # user_data = <<-EOF
  #             #!/bin/bash
  #             sudo dnf update -y
  #             sudo dnf install -y git python3 python3-pip tmux spal-release
  #             sudo dnf install -y podman
  #             curl -LsSf https://astral.sh/uv/install.sh | sh
  #             source "$HOME/.local/bin/env"
  #             echo "EC2 ready for CBI processing" > /var/log/cbi-setup.log
  #             EOF

  tags = {
    Name        = "cbi-composite-burn-index"
    Purpose     = "landsat-processing"
    Environment = "dev"
  }
}

# Outputs
output "instance_public_ip" {
  value       = aws_instance.cbi_compute.public_ip
  description = "SSH into the instance using this IP"
}

output "ssh_command" {
  value       = "ssh ec2-user@${aws_instance.cbi_compute.public_ip}"
  description = "Example SSH command (after adding your key)"
}
