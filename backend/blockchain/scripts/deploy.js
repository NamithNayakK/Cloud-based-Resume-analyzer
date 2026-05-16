const hre = require('hardhat');

async function main() {
  const ResumeVerification = await hre.ethers.getContractFactory('ResumeVerification');
  const contract = await ResumeVerification.deploy();
  await contract.deployed();

  console.log('ResumeVerification deployed to:', contract.address);
}

main()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error(error);
    process.exit(1);
  });
