const { ethers } = require('ethers');

const BLOCKCHAIN_RPC_URL = process.env.BLOCKCHAIN_RPC_URL || 'http://127.0.0.1:8545';
const CONTRACT_ADDRESS = process.env.RESUME_VERIFICATION_CONTRACT_ADDRESS || '';
const PRIVATE_KEY = process.env.PRIVATE_KEY || '';

const RESUME_VERIFICATION_ABI = [
  'function storeResumeHash(bytes32 hash) public returns (bool)',
  'function verifyResumeHash(bytes32 hash) public view returns (bool)',
  'event ResumeHashStored(bytes32 indexed hash, address indexed uploader, uint256 timestamp)',
];

let provider;
let signer;
let contract;

function initBlockchainClient() {
  if (provider) {
    return;
  }

  provider = new ethers.providers.JsonRpcProvider(BLOCKCHAIN_RPC_URL);

  if (PRIVATE_KEY) {
    signer = new ethers.Wallet(PRIVATE_KEY, provider);
  }

  if (CONTRACT_ADDRESS) {
    contract = new ethers.Contract(CONTRACT_ADDRESS, RESUME_VERIFICATION_ABI, signer || provider);
  }
}

async function storeResumeHash(hash) {
  initBlockchainClient();

  if (!contract) {
    return {
      success: false,
      message: 'Blockchain contract not configured. Set RESUME_VERIFICATION_CONTRACT_ADDRESS and run the local contract deployment.',
      hash,
    };
  }

  try {
    const tx = await contract.storeResumeHash(hash);
    const receipt = await tx.wait();
    return {
      success: receipt.status === 1,
      transactionHash: receipt.transactionHash,
      blockNumber: receipt.blockNumber,
      hash,
    };
  } catch (error) {
    return {
      success: false,
      message: error.message || 'Failed to store hash on blockchain.',
      hash,
    };
  }
}

async function verifyResumeHash(hash) {
  initBlockchainClient();

  if (!contract) {
    return {
      verified: false,
      message: 'Blockchain contract not configured. Set RESUME_VERIFICATION_CONTRACT_ADDRESS.',
      hash,
    };
  }

  try {
    const verified = await contract.verifyResumeHash(hash);
    return {
      verified,
      hash,
    };
  } catch (error) {
    return {
      verified: false,
      message: error.message || 'Blockchain verification failed.',
      hash,
    };
  }
}

module.exports = {
  storeResumeHash,
  verifyResumeHash,
};
