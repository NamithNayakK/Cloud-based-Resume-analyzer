# Blockchain Verification Module

This folder contains the smart contract and deployment scripts needed to verify resume hashes on a local blockchain.

## Setup

1. Change into the blockchain folder:

   ```bash
   cd backend/blockchain
   ```

2. Install Hardhat dependencies:

   ```bash
   npm install
   ```

3. Compile the contract:

   ```bash
   npm run compile
   ```

4. Deploy to the Hardhat local network:

   ```bash
   npm run deploy
   ```

The script prints the deployed contract address. Add that address to `backend/.env` as `RESUME_VERIFICATION_CONTRACT_ADDRESS`.

## Smart contract

- `contracts/ResumeVerification.sol` stores SHA-256 resume hashes.
- `scripts/deploy.js` deploys the contract to the local Hardhat network.

## Notes

- Use `BLOCKCHAIN_RPC_URL` to point your backend to a local or remote JSON-RPC provider.
- Use `PRIVATE_KEY` to sign transactions for hash storage.
