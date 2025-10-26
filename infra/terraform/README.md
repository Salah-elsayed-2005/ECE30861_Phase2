git fetch origin
git reset --hard origin/main   # throw away the rejected local commit(s)

# make sure .gitignore blocks terraform artifacts
Add-Content .gitignore ".terraform/"
Add-Content .gitignore "*.tfstate"
Add-Content .gitignore "*.tfstate.backup"
Add-Content .gitignore "crash.log"
Add-Content .gitignore "*.tfvars"

# stage ONLY the good infra files
git add .gitignore infra/terraform/*.tf infra/terraform/README.md infra/terraform/.terraform.lock.hcl infra/deploy.ps1