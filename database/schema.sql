-- Esquema Eficiencia2D — compatible con ERD Editor (PostgreSQL)
-- Nota: ON DELETE no se incluye aquí; el parser de ERD Editor no lo soporta en CREATE TABLE.
-- En producción SQLAlchemy usa RESTRICT (salvo configuraciones_usuario → CASCADE).

CREATE TABLE usuarios (
    id UUID PRIMARY KEY,
    email VARCHAR NOT NULL UNIQUE,
    password_hash VARCHAR NOT NULL,
    nombre VARCHAR,
    fecha_creacion TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    estado VARCHAR NOT NULL DEFAULT 'activo',
    email_verification_token VARCHAR(64) UNIQUE,
    email_verified_at TIMESTAMPTZ
);

CREATE TABLE planes (
    id SERIAL PRIMARY KEY,
    nombre VARCHAR NOT NULL,
    precio NUMERIC(10, 2) NOT NULL,
    limite_almacenamiento_mb INTEGER,
    limite_proyectos INTEGER
);

CREATE TABLE suscripciones (
    id UUID PRIMARY KEY,
    usuario_id UUID NOT NULL UNIQUE,
    plan_id INTEGER NOT NULL,
    estado VARCHAR NOT NULL,
    fecha_inicio TIMESTAMPTZ NOT NULL,
    fecha_fin TIMESTAMPTZ NOT NULL,
    proveedor_pago_id VARCHAR,
    CONSTRAINT fk_suscripciones_usuario FOREIGN KEY (usuario_id) REFERENCES usuarios (id),
    CONSTRAINT fk_suscripciones_plan FOREIGN KEY (plan_id) REFERENCES planes (id)
);

CREATE TABLE proyectos (
    id UUID PRIMARY KEY,
    usuario_id UUID NOT NULL,
    nombre VARCHAR NOT NULL,
    formato VARCHAR NOT NULL,
    tamano_bytes BIGINT NOT NULL,
    url_archivo VARCHAR NOT NULL,
    metadata_impresion JSONB,
    fecha_creacion TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT fk_proyectos_usuario FOREIGN KEY (usuario_id) REFERENCES usuarios (id)
);

CREATE TABLE pagos (
    id UUID PRIMARY KEY,
    usuario_id UUID NOT NULL,
    suscripcion_id UUID NOT NULL,
    monto NUMERIC(10, 2) NOT NULL,
    moneda VARCHAR(3) NOT NULL,
    estado VARCHAR NOT NULL,
    pasarela_pago VARCHAR,
    transaccion_externa_id VARCHAR NOT NULL UNIQUE,
    fecha_pago TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT fk_pagos_usuario FOREIGN KEY (usuario_id) REFERENCES usuarios (id),
    CONSTRAINT fk_pagos_suscripcion FOREIGN KEY (suscripcion_id) REFERENCES suscripciones (id)
);

CREATE TABLE configuraciones_usuario (
    id UUID PRIMARY KEY,
    usuario_id UUID NOT NULL UNIQUE,
    tema_color VARCHAR NOT NULL DEFAULT 'oscuro',
    idioma VARCHAR(5) NOT NULL DEFAULT 'es',
    notificaciones_email BOOLEAN NOT NULL DEFAULT TRUE,
    preferencias_interfaz JSONB,
    CONSTRAINT fk_configuraciones_usuario FOREIGN KEY (usuario_id) REFERENCES usuarios (id)
);

CREATE TABLE cupones (
    id UUID PRIMARY KEY,
    codigo VARCHAR NOT NULL UNIQUE,
    descripcion VARCHAR,
    limite_usos INTEGER NOT NULL DEFAULT 1,
    limite_usos_por_usuario INTEGER NOT NULL DEFAULT 1,
    plan_id INTEGER,
    descuento_porcentaje NUMERIC(5, 2),
    descuento_monto NUMERIC(10, 2),
    fecha_inicio TIMESTAMPTZ,
    fecha_expiracion TIMESTAMPTZ,
    activo BOOLEAN NOT NULL DEFAULT TRUE,
    limitaciones JSONB,
    fecha_creacion TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT fk_cupones_plan FOREIGN KEY (plan_id) REFERENCES planes (id)
);

CREATE TABLE usos_cupon (
    id UUID PRIMARY KEY,
    cupon_id UUID NOT NULL,
    usuario_id UUID NOT NULL,
    fecha_uso TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT fk_usos_cupon_cupon FOREIGN KEY (cupon_id) REFERENCES cupones (id),
    CONSTRAINT fk_usos_cupon_usuario FOREIGN KEY (usuario_id) REFERENCES usuarios (id)
);

CREATE INDEX ix_usos_cupon_cupon_usuario ON usos_cupon (cupon_id, usuario_id);
