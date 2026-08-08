-- -------------------------------------------------------------------------------------------
-- SCRIPT de Creación de la BBDD y TABLAS para el proyecto Chess Rekognition.
-- --------------------------------------------------------------------------------------------

-- TABLA DE USUARIOS
CREATE TABLE usuarios 
(
    username    VARCHAR(50)     PRIMARY KEY,
    nombre      VARCHAR(255)    NOT NULL,
    apellidos   VARCHAR(255)    NOT NULL,
    password    VARCHAR(255)    NOT NULL, 
    mail        VARCHAR(255)    NOT NULL
);

-- TABLA DE PARTIDAS
CREATE TABLE partidas 
(
    id_partida     INT              AUTO_INCREMENT  PRIMARY KEY,
    username       VARCHAR(50)      NOT NULL,
    evento         VARCHAR(250)     NOT NULL,
    blancas        VARCHAR(250)     NOT NULL,
    negras         VARCHAR(250)     NOT NULL,
    fecha          DATE             NOT NULL,
    resultado      VARCHAR(7)       NOT NULL,
    pgn            LONGTEXT         NOT NULL,
    tipo_partida   VARCHAR(2)       DEFAULT NULL,
    ronda          INT,
    tablero        INT,
    lugar          VARCHAR(250),
    observaciones  TEXT,
    fecha_registro DATETIME         NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (username) REFERENCES usuarios(username)
);

-- TABLA DE RETRANSMISIONES
CREATE TABLE retransmisiones 
(
    id_retransmision    INT             AUTO_INCREMENT  PRIMARY KEY,
    token               VARCHAR(64)     NOT NULL UNIQUE,
    username            VARCHAR(50)     NOT NULL,
    evento              VARCHAR(250)    NOT NULL,
    blancas             VARCHAR(250)    NOT NULL,
    negras              VARCHAR(250)    NOT NULL,
    resultado           VARCHAR(7)      DEFAULT NULL,
    fen                 TEXT            DEFAULT NULL,
    pgn                 LONGTEXT        DEFAULT NULL,
    ronda          		INT,
    tablero        		INT,
    lugar          		VARCHAR(250),
    is_activa         	BOOLEAN         NOT NULL DEFAULT FALSE, 
    fecha_creacion      DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP,
    fecha_actualizacion DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (username) REFERENCES usuarios(username)
);

-- INSERT
INSERT INTO usuarios (username, nombre, apellidos, password, mail)
VALUES (
    'chess-test01',
    'José Joaquín',
    'Sánchez Fernández',
    '$2b$12$HFsVcKAxRWs6ATHXO1MJR.MjDMTC/d5HqdTz7Uyo1xWZYR5RNPghC',  -- chess-test01
    'web@ajedrezcoimbra.com
);
