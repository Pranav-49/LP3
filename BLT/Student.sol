// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract StudentData 
{
    struct Student 
    {
        uint256 rollNo;
        string name;
        uint256 age;
    }

    Student[] public students;

    event StudentAdded(uint256 rollNo, string name, uint256 age);

    function addStudent(uint256 _rollNo, string memory _name, uint256 _age) public 
    {
        students.push(Student(_rollNo, _name, _age));
        emit StudentAdded(_rollNo, _name, _age);
    }

    function getStudent(uint256 index) public view returns (uint256, string memory, uint256) 
    {
        require(index < students.length, "Invalid index");
        Student memory s = students[index];
        return (s.rollNo, s.name, s.age);
    }

    function getAllStudents() public view returns (Student[] memory) 
    {
        return students;
    }

    event LogFallback(string message);
    event LogReceive(string message);

    fallback() external payable {
        emit LogFallback("Fallback function executed");
    }

    receive() external payable {
        emit LogReceive("Receive function executed");
    }
}
