// SPDX-License-Identifier: UNLICENSED
pragma solidity ^0.8.0;

contract StudentData {

    struct Student {
        string name;
        uint rollno;
    }

    Student[] public studentArr;

    function addStudent(string memory name, uint rollno) public {
        for (uint i = 0; i < studentArr.length; i++) {
            if (studentArr[i].rollno == rollno) {
                revert("Student with roll no already exists!");
            }
        }
        studentArr.push(Student(name, rollno));
    }

    function displayAllStudents() public view returns (Student[] memory) {
        return studentArr;
    }


    function getLengthOfStudents() public view returns (uint) {
        return studentArr.length;
    }

    event FundsReceived(address sender, uint amount);
    event FallbackTriggered(address sender, uint amount, bytes data);

    receive() external payable {
        emit FundsReceived(msg.sender, msg.value);
    }

    fallback() external payable {
        emit FallbackTriggered(msg.sender, msg.value, msg.data);
    }
}
