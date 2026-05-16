// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

contract ResumeVerification {
    mapping(bytes32 => bool) private resumeHashes;

    event ResumeHashStored(bytes32 indexed hash, address indexed uploader, uint256 timestamp);

    function storeResumeHash(bytes32 hash) external returns (bool) {
        require(hash != bytes32(0), "Invalid hash");
        resumeHashes[hash] = true;
        emit ResumeHashStored(hash, msg.sender, block.timestamp);
        return true;
    }

    function verifyResumeHash(bytes32 hash) external view returns (bool) {
        return resumeHashes[hash];
    }
}
