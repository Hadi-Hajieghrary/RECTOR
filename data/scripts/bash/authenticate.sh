#!/bin/bash
# Helper script to authenticate with Google Cloud for Waymo dataset access
# Usage: ./authenticate.sh

set -e

echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║  Google Cloud Authentication for Waymo Dataset                    ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo "ERROR: gcloud CLI not installed"
    echo "Install with: curl https://sdk.cloud.google.com | bash"
    exit 1
fi

echo "This will authenticate you with Google Cloud in two steps:"
echo ""
echo "Step 1: User authentication (for gsutil access)"
echo "Step 2: Application default credentials (for Waymo API)"
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 0
fi

echo ""
echo "════════════════════════════════════════════════════════════════════"
echo "Step 1: User Authentication"
echo "════════════════════════════════════════════════════════════════════"
echo ""
echo "You'll receive a URL. Open it in a browser, sign in with your Google"
echo "account, and paste the verification code back here."
echo ""
read -p "Press Enter to continue..."

gcloud auth login --no-browser || {
    echo ""
    echo "ERROR: User authentication failed"
    exit 1
}

echo ""
echo "✓ Step 1 complete!"
echo ""
echo "════════════════════════════════════════════════════════════════════"
echo "Step 2: Application Default Credentials"
echo "════════════════════════════════════════════════════════════════════"
echo ""
read -p "Press Enter to continue..."

gcloud auth application-default login --no-browser || {
    echo ""
    echo "WARNING: Application-default credentials setup failed"
    echo "This may not be required for dataset download."
}

# Get authenticated account once
active_account=$(gcloud auth list --filter=status:ACTIVE --format='value(account)' 2>/dev/null | head -1)

echo ""
echo "════════════════════════════════════════════════════════════════════"
echo "✓ Authentication Complete!"
echo "════════════════════════════════════════════════════════════════════"
echo ""
echo "Authenticated as: $active_account"
echo ""
echo "IMPORTANT: Accept the Waymo Open Dataset license at:"
echo "  https://waymo.com/open/terms"
echo ""
echo "Now you can run:"
echo "  ./waymo download --num 5"
echo ""
