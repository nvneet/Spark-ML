CREATE DATABASE bookapi;

USE bookapi;

CREATE TABLE book(
	id INT PRIMARY KEY AUTO_INCREMENT,
    title VARCHAR(255) NOT NULL,
    author VARCHAR(255) NOT NULL
);

SELECT * FROM book;