//SPDX-License-Identifier: BankSystem

pragma solidity ^0.8.26;


contract SimpleBank 
{
    address public owner;
    uint256 public balance;

    constructor() 
    {
        owner = msg.sender;
        balance = 0;
    }

    function deposit(uint256 amount) public payable
    {
        require(msg.sender == owner, "Only the owner can deposit money");
        balance += amount;
    }

    function withdraw(uint256 amount) public payable
    {
        require(msg.sender == owner, "Only the owner can withdraw money");
        require(amount <= balance, "Insufficient balance");
        balance -= amount;
    }

    function getBalance() public view returns (uint256) {
        return balance;
    }
}