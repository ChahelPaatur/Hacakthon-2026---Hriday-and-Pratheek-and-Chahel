# Apple Code Signing & Notarization Setup for NeuroLang

Without signing and notarization, macOS Gatekeeper blocks downloaded
binaries with: *"can't be opened because Apple cannot check it for
malicious software."*

This guide walks through getting a signed + notarized macOS binary via
the GitHub Actions release workflow.

---

## Prerequisites

| Requirement | Notes |
|---|---|
| Apple Developer Account | $99/year — [developer.apple.com](https://developer.apple.com) |
| macOS machine | Needed once to export the certificate |
| GitHub repo admin | To add repository secrets |

You do **not** need Xcode.app — just the Command Line Tools
(`xcode-select --install`).

---

## Step 1: Create a Developer ID Application Certificate

1. Go to [Apple Developer > Certificates](https://developer.apple.com/account/resources/certificates/list)
2. Click **"+"** to create a new certificate
3. Select **"Developer ID Application"** and click Continue
4. Generate a **Certificate Signing Request (CSR)**:
   - Open **Keychain Access** on your Mac
   - Menu: Keychain Access > Certificate Assistant > Request a Certificate From a Certificate Authority
   - Enter your email, select "Saved to disk", click Continue
5. Upload the CSR file to Apple and download the certificate
6. Double-click the `.cer` file to install it into your Keychain

## Step 2: Export the Certificate as .p12

1. Open **Keychain Access**
2. In the sidebar, click **"My Certificates"** (important — not "Certificates")
3. Find your **"Developer ID Application: Your Name (TEAM_ID)"** entry
4. Expand it to confirm the private key is attached
5. Right-click the certificate → **Export**
6. Save as `.p12` format
7. Set a strong password — you'll need this as a GitHub secret

## Step 3: Base64-Encode the Certificate

```bash
base64 -i ~/path/to/certificate.p12 | pbcopy
```

This copies the base64 string to your clipboard.

## Step 4: Create an App-Specific Password

1. Go to [appleid.apple.com](https://appleid.apple.com)
2. Sign In > Security > App-Specific Passwords
3. Click **"+"** to generate one, name it "NeuroLang CI"
4. Copy the generated password

## Step 5: Add GitHub Repository Secrets

Go to your repo: **Settings > Secrets and variables > Actions > New repository secret**

Add these 5 secrets:

| Secret Name | Value |
|---|---|
| `APPLE_CERTIFICATE_BASE64` | The base64 string from Step 3 |
| `APPLE_CERTIFICATE_PASSWORD` | The .p12 export password from Step 2 |
| `APPLE_SIGNING_IDENTITY` | Full name from Keychain, e.g. `Developer ID Application: Your Name (TEAMID)` |
| `APPLE_ID` | Your Apple ID email |
| `APPLE_TEAM_ID` | Your 10-character team ID (visible in Apple Developer portal) |
| `APPLE_APP_SPECIFIC_PASSWORD` | The app-specific password from Step 4 |

## Step 6: Trigger a Release

```bash
git tag v1.0.0
git push origin v1.0.0
```

The workflow will:
1. Build the macOS binary (both ARM and Intel)
2. Sign it with your Developer ID certificate
3. Submit it to Apple for notarization (waits for approval)
4. Upload the signed + notarized binary to GitHub Releases

---

## How It Works in the Workflow

The release workflow has three macOS signing stages:

1. **Certificate Import**: Creates a temporary keychain in CI, imports
   the .p12 certificate
2. **Code Signing**: Signs with `codesign --sign "Developer ID..." --timestamp --options runtime`
3. **Notarization**: Submits to Apple via `xcrun notarytool submit --wait`

If the signing secrets aren't set, the workflow falls back to an ad-hoc
signature (which still won't pass Gatekeeper, but lets the build succeed).

---

## Verification

After downloading a release binary, verify it's properly signed:

```bash
codesign --verify --deep --strict --verbose=2 neurolang-macos-arm64
spctl --assess --verbose=2 neurolang-macos-arm64
```

Both commands should report the binary as valid and accepted.

---

## Cost

- Apple Developer Program: **$99/year**
- Notarization: **Free** (included with Developer Program)
- No Xcode required — Command Line Tools only
