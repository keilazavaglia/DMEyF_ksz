# Perfiles de clientes que dejan el banco (BAJA+2) — 4 grupos

**Fuente:** `competencia_01.csv`, meses 202103–202106 (julio y agosto 2021 no tienen BAJA+2 y se excluyen).
**Método:** Random Forest entrenado para distinguir 4.053 clientes BAJA+2 de 4.053 clientes fieles → cada cliente BAJA+2 queda descripto por *qué variables usó el bosque para reconocerlo* → KMeans con k=4 sobre ese perfil.
**Por qué 4 grupos:** los grupos son muy estables (ARI 0,98: casi no cambian si se cambia la muestra) y el más chico tiene el 17% de los clientes. Con 5 se separa un grupo de solo 5%. Con 6 o más, los grupos pierden estabilidad.

Todos los números son la **mediana** (o el % de clientes) en el **mes en que el cliente es BAJA+2**, es decir, dos meses antes de irse: es el momento en que todavía se puede actuar. Se comparan contra los clientes fieles en los mismos meses.

---

## Resumen de los 4 grupos

| | 1. Solo usa la tarjeta | 2. Se muda a otro banco | 3. Nunca se enganchó | 4. Sin tarjetas y en rojo | Fieles |
|---|---|---|---|---|---|
| Clientes (% de las bajas) | 1.356 (33%) | 1.122 (28%) | 877 (22%) | 698 (17%) | — |
| Antigüedad (meses) | 101 | 108 | **57** | 78 | 127 |
| Movimientos en el trimestre | 26 | 74 | **5** | **3** | 106 |
| Cobra el sueldo en el banco | 1% | **26%** | 1% | 0% | 55% |
| Saldo total de cuentas | −$898 | $12.841 | −$3.881 | **−$10.358** | $37.915 |
| Compras con Visa en el mes | **3** | 2 | 0 | 0 | 9 |
| Sin tarjeta Visa | 4% | 13% | 12% | **97%** | 5% |
| Tiene préstamo personal | 8% | 9% | 7% | **36%** | 22% |
| Paga comisión de mantenimiento | 69% | 38% | 73% | 69% | 30% |
| Usa la app en el mes | 49% | **73%** | 23% | 18% | 74% |
| Rentabilidad anual para el banco | $8.280 | $9.835 | $9.041 | **$17.582** | $13.019 |

---

## Grupo 1 — "Solo usa la tarjeta" (1.356 clientes, 33%)

**Variables que lo definen (las que más usa el modelo para reconocerlo):**
- `ctarjeta_visa_transacciones`: cantidad de compras con Visa en el mes.
- `mtarjeta_visa_consumo`: monto consumido con Visa.
- `mttarjeta_visa_debitos_automaticos`: débitos automáticos adheridos a la Visa.

**Características**
- Es un cliente antiguo (8 años) y sigue activo (98% hizo movimientos voluntarios en el trimestre).
- Su relación con el banco es casi solo la tarjeta de crédito. Casi nadie cobra el sueldo acá (1% contra 55% de los fieles) y la caja de ahorro está prácticamente vacía ($226).
- Usa la tarjeta mucho menos que un fiel: 3 compras contra 9, y $6.076 de consumo contra $19.809.
- El 69% paga comisión de mantenimiento del paquete: paga por un paquete que casi no usa.
- Mora algo mayor que el resto (6% contra 2% de los fieles).

**Señal de alerta:** el consumo con Visa baja mes a mes ($7.844 → $6.076 en los 3 meses previos) y el saldo de la cuenta pasa de positivo a negativo.

**Medidas**
- Reactivar la tarjeta: cashback, cuotas sin interés o promociones en los rubros donde ya compra.
- Bonificar la comisión de mantenimiento durante un tiempo, o pasarlo a un paquete más barato.
- Ofrecer un beneficio por acreditar el sueldo en el banco: es lo que más diferencia a este grupo de un cliente fiel.
- Disparar la acción cuando el consumo con tarjeta cae dos meses seguidos.

---

## Grupo 2 — "Se muda a otro banco" (1.122 clientes, 28%)

**Variables que lo definen:**
- `mpayroll` / `cpayroll_trx`: monto y cantidad de acreditaciones de sueldo.
- `matm`: monto operado en cajeros del banco.

**Características**
- Es el grupo que más se parece a un fiel: antiguo (9 años), digital (73% usa la app y hace 20 operaciones por home banking), con transferencias, débito automático y algún plazo fijo.
- Tiene saldo positivo ($12.841) y no paga comisión de mantenimiento.
- La diferencia es que **está perdiendo la acreditación del sueldo**: solo el 26% la sigue cobrando acá, contra el 55% de los fieles.

**Señal de alerta:** la mudanza se ve en los datos. En los 3 meses previos:
- los que cobran sueldo acá bajan de 37% a 26%;
- el saldo cae de $31.936 a $12.841;
- la caja de ahorro baja de $18.609 a $12.269;
- las compras con tarjeta bajan de $9.269 a $3.975.

Están vaciando la cuenta y llevándose el sueldo a otro lado.

**Medidas**
- **Alerta temprana:** si un mes no se acredita el sueldo, o el saldo cae más de un 30%, contactar al cliente en ese momento. Es el grupo con la ventana de acción más clara.
- Ofrecer retención por portabilidad: mejor tasa de plazo fijo, beneficio por volver a acreditar el sueldo, igualar la oferta de la competencia.
- Es el grupo más valioso para retener: digital y con plata en el banco, se parece al cliente ideal.

---

## Grupo 3 — "Nunca se enganchó" (877 clientes, 22%)

**Variables que lo definen:**
- `active_quarter`: si hizo movimientos voluntarios en el trimestre. Solo el 60% lo hizo, contra el 99% de los fieles.
- `Visa_mfinanciacion_limite`: límite de financiación de la Visa (más bajo que el de los fieles).
- `Visa_fultimo_cierre`: días desde el último cierre de la tarjeta.

**Características**
- Es el grupo más nuevo: 57 meses de antigüedad, contra 127 de los fieles.
- Tiene los productos (6 de mediana; el 88% tiene Visa) pero **no los usa**: 5 movimientos en el trimestre, 0 compras con tarjeta, 0 operaciones por home banking, 23% de uso de la app.
- No deja plata en el banco (el margen por depósitos es prácticamente $0) y el 74% tiene saldo negativo.
- Igual paga comisión de mantenimiento (73%): solo ve el costo del banco, no el beneficio.

**Señal de alerta:** el saldo negativo crece mes a mes (−$742 → −$3.881). La inactividad viene de antes: es una baja anunciada.

**Medidas**
- Hacer onboarding y activación en los primeros años: beneficio por la primera compra con tarjeta, por adherir débitos automáticos o por bajar la app.
- Bonificar el paquete mientras no lo usa, o pasarlo a una cuenta sin costo. Así no se va por la comisión y se puede venderle más adelante.
- Si la activación no funciona, evaluar si conviene retenerlo: es el grupo con menor vínculo.

---

## Grupo 4 — "Sin tarjetas y en rojo" (698 clientes, 17%)

**Variables que lo definen:**
- `cdescubierto_preacordado`: si tiene acuerdo de descubierto (56%, contra 97% de los fieles).
- `cprestamos_personales`: tiene préstamo personal (36%, contra 22% de los fieles).
- `internet`: uso de home banking o app (muy bajo).

**Características**
- **El 95% ya no tiene ninguna tarjeta de crédito.** No es que las tarjetas se cierren en estos meses: tres meses antes ya no las tenía.
- Es el grupo más endeudado: el 84% tiene saldo negativo (mediana −$10.358) y un tercio tiene préstamo personal (mediana $100.399).
- Casi no opera: 3 movimientos en el trimestre, sin home banking.
- **Es el más rentable para el banco** ($17.582 al año, contra $13.019 de los fieles), pero por intereses: el banco le gana por la deuda, no por su actividad.

**Señal de alerta:** el saldo negativo se profundiza (−$7.022 → −$10.358) y tiene cada vez menos productos (5, contra 7 de los fieles).

**Medidas**
- Ofrecer reestructurar la deuda: consolidar el descubierto en un préstamo a menor tasa o en cuotas. Así se reduce el riesgo y se mantiene la relación.
- Antes de que termine de pagar el préstamo, ofrecer uno nuevo preaprobado.
- Volver a ofrecerle una tarjeta, con límite bajo o garantizada, para recuperar uso cotidiano.
- Es prioritario por rentabilidad: perder a uno de estos clientes cuesta más que perder a uno de cualquier otro grupo.

---

## Guion para video de 5 minutos

> Unas 700 palabras, ritmo de habla normal. Entre corchetes, qué mostrar en pantalla.

**[0:00–0:40] Introducción** — *[título + número de bajas por mes]*
"Cada mes se van del banco alrededor de mil clientes del paquete premium. La buena noticia es que se puede saber quiénes se van a ir con dos meses de anticipación: son los que marcamos como BAJA+2. La pregunta es si todos se van por lo mismo. Analizamos los 4.053 clientes BAJA+2 de marzo a junio de 2021 y la respuesta es no: hay cuatro perfiles bien distintos, y cada uno necesita una medida diferente."

**[0:40–1:10] Cómo los encontramos** — *[gráfico k_eval.png]*
"Entrenamos un modelo para distinguir a los que se van de los que se quedan. Después agrupamos a los que se van según qué variables usó el modelo para reconocerlos. Probamos de 2 a 8 grupos. Con 4, los grupos son estables y todos tienen un tamaño relevante. Con más de 5, empiezan a cambiar según la muestra."

**[1:10–2:00] Grupo 1: "Solo usa la tarjeta" (33%)** — *[gráfico de mtarjeta_visa_consumo]*
"El grupo más grande. Son clientes antiguos y activos, pero su relación con el banco es solo la tarjeta de crédito: casi ninguno cobra el sueldo acá y la caja de ahorro está vacía. Además, la usan cada vez menos: 3 compras al mes contra 9 de un cliente fiel. Y siete de cada diez pagan comisión de mantenimiento por un paquete que no aprovechan. La medida: reactivar la tarjeta con cashback y cuotas, bonificar la comisión, y darles un motivo para traer el sueldo."

**[2:00–2:50] Grupo 2: "Se muda a otro banco" (28%)** — *[gráfico de mpayroll y mcuentas_saldo]*
"Este es el que más duele, porque se parece al cliente ideal: digital, con transferencias, débitos automáticos y plata en la cuenta. Pero en los meses previos se ve la mudanza: los que cobran el sueldo acá bajan del 37 al 26%, y el saldo cae de 32 mil a 13 mil pesos. Se están llevando el sueldo y los ahorros a otro banco. La medida es una alerta temprana: si un mes no entra el sueldo o el saldo cae fuerte, hay que llamar a ese cliente en ese momento con una oferta de retención."

**[2:50–3:40] Grupo 3: "Nunca se enganchó" (22%)** — *[gráfico de active_quarter / ctrx_quarter]*
"Son los más nuevos: menos de cinco años en el banco, contra más de diez de los fieles. Tienen los productos pero no los usan: 5 movimientos en tres meses, ninguna compra con tarjeta, casi no usan la app. Lo único que ven del banco es la comisión mensual. La medida es de onboarding: incentivos para la primera compra, para adherir débitos y bajar la app, y no cobrarles el paquete mientras no lo usen."

**[3:40–4:30] Grupo 4: "Sin tarjetas y en rojo" (17%)** — *[ficha del cluster: saldo y rentabilidad]*
"El grupo más chico, pero el más rentable: deja 17 mil pesos al año contra 13 mil de un fiel. Lo que pasa es que esa ganancia viene de los intereses. El 95% ya no tiene tarjeta de crédito, el 84% está en saldo negativo y un tercio tiene un préstamo personal. Casi no opera. La medida: ofrecer reestructurar la deuda en cuotas a mejor tasa, un préstamo nuevo preaprobado antes de que termine el actual, y volver a darles una tarjeta para que usen el banco todos los días."

**[4:30–5:00] Cierre** — *[tabla resumen de los 4 grupos]*
"En resumen: no todos los que se van se van por lo mismo. Uno solo usa la tarjeta, otro se muda de banco, otro nunca se enganchó y otro está endeudado. Con dos meses de anticipación y el perfil correcto, cada uno puede recibir la medida que le sirve, en vez de una campaña genérica para todos."

---

### Notas para quien arme el video
- Los montos son pesos nominales de 2021. Sirven para comparar entre grupos, no como valores absolutos.
- Los gráficos de tendencia de cada variable están en `clusters_tendencias.pdf`. La página 2 es la ficha de los 4 grupos contra los fieles.
- La asignación cliente → grupo está en `clusters_asignacion.csv`.
