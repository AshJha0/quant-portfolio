-- Small, hand-written data set for the SQL pack. Times are seconds after the open.
CREATE TABLE trades (id INTEGER PRIMARY KEY, sym TEXT, t INTEGER, px REAL, qty INTEGER, side TEXT);
CREATE TABLE quotes (sym TEXT, t INTEGER, bid REAL, ask REAL);
CREATE TABLE closes (sym TEXT, d TEXT, close REAL);

INSERT INTO trades VALUES
 (1,'AAA',  5,10.00,100,'B'),(2,'AAA', 12,10.05,200,'S'),(3,'BBB', 13,50.00, 50,'B'),
 (4,'AAA', 20,10.02,300,'B'),(5,'BBB', 25,50.10,100,'S'),(6,'AAA', 31,10.10,100,'S'),
 (7,'BBB', 40,49.90,150,'B'),(8,'AAA', 47,10.08,400,'B'),(9,'CCC', 50, 5.00,1000,'B'),
 (10,'BBB',58,50.20, 50,'S');
INSERT INTO quotes VALUES
 ('AAA', 0, 9.99,10.01),('AAA',10,10.03,10.06),('AAA',30,10.08,10.11),('AAA',45,10.07,10.09),
 ('BBB', 0,49.95,50.05),('BBB',20,50.05,50.15),('BBB',35,49.85,49.95),
 ('CCC',55, 4.99, 5.01);
INSERT INTO closes VALUES
 ('AAA','2026-01-05',10.0),('AAA','2026-01-06',10.2),('AAA','2026-01-07',10.1),
 ('AAA','2026-01-08',10.4),('AAA','2026-01-09',10.3),
 ('BBB','2026-01-05',50.0),('BBB','2026-01-06',49.0),('BBB','2026-01-07',49.5),
 ('BBB','2026-01-09',51.0);
