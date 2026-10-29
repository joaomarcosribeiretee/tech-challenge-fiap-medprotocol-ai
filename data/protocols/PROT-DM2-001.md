---
protocol_id: PROT-DM2-001
titulo: Acompanhamento ambulatorial de adultos com diabetes mellitus tipo 2
versao: 1.0
vigencia: 2026-01-01
status: SIMULACAO
natureza_regras: simulada
referencias_externas_verificadas: nenhuma
validacao_clinica: nao_realizada
escopo: Hospital fictício Santa Aurora - uso exclusivamente educacional
---

# PROT-DM2-001 — Acompanhamento ambulatorial de adultos com diabetes mellitus tipo 2

> SIMULAÇÃO. Protocolo fictício criado para fins educacionais no projeto MedProtocol AI.
> Não é diretriz médica oficial, não foi validado clinicamente e não deve orientar o cuidado
> de pacientes reais. Todos os valores numéricos são ilustrativos.

## [S1] Objetivo

Padronizar, no hospital fictício Santa Aurora, a periodicidade dos exames laboratoriais e a
sinalização de verificações pendentes no acompanhamento ambulatorial de adultos com diabetes
mellitus tipo 2.

## [S2] Critérios de inclusão

Adultos com diagnóstico registrado de diabetes mellitus tipo 2 em acompanhamento ambulatorial
no hospital fictício Santa Aurora.

## [S3] Periodicidade dos exames

- A meta institucional simulada de HbA1c é resultado menor ou igual a 7,0%.
- HbA1c com último resultado acima de 7,0%: repetir em 90 dias.
- HbA1c com último resultado menor ou igual a 7,0%: repetir em 180 dias.
- Creatinina sérica e TFG estimada: repetir a cada 365 dias.
- LDL-colesterol: repetir a cada 365 dias.
- O limite institucional simulado de LDL-colesterol é resultado abaixo de 100 mg/dL.

## [S4] Situação das verificações

- A situação de cada exame é avaliada em relação a uma data de referência.
- Próxima verificação prevista: data de coleta do último resultado somada ao intervalo do exame.
- Em dia: a próxima verificação prevista é igual ou posterior à data de referência.
- Vencido: existe resultado registrado, mas a próxima verificação prevista é anterior à data de referência.
- Ausente: não existe nenhum resultado registrado do exame.
- Resultado fora da meta ou do limite é avaliado independentemente do prazo: um exame pode estar em dia e fora da meta.
- Exame vencido ou ausente deve ser sinalizado como verificação pendente.
- Valores de exames vencidos ou ausentes nunca devem ser estimados ou inferidos.

## [LIM] Limites de utilização

- Este protocolo é uma simulação e não substitui avaliação profissional.
- Não define diagnóstico, tratamento, medicamento ou dose.
- Não deve ser aplicado a pacientes reais.

## [REV] Encaminhamento para revisão humana

- Qualquer sugestão de início, ajuste ou suspensão de medicamento ou dose deve ser validada pelo médico responsável.
- Resultados fora da meta ou do limite simulado devem ser apresentados ao médico responsável para decisão de conduta.
- Solicitações de prescrição devem ser recusadas pelo assistente e encaminhadas ao médico responsável.
- Dados conflitantes ou incompletos devem ser sinalizados ao profissional responsável.
