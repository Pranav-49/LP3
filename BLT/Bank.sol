//SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract Bank {
    uint256 balence = 0;

    function withdrawal() public payable{
        require(msg.value < balence, "Insuffiecent balence");
        balence -= msg.value;
    } 

    function deposite() public payable{
        require(msg.value > 0, "Amount sould be greater than zero");
        balence += msg.value;
    }

    function show() public view returns(uint){
        return balence;
    }
}