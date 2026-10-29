# Instruções para agentes de desenvolvimento

Este repositório contém o projeto MedProtocol AI, desenvolvido para o Tech Challenge Fase 3 da FIAP.

Leia `docs/PROJECT_SPEC.md` antes de implementar qualquer funcionalidade.

## Regras de trabalho

1. Trabalhe em incrementos pequenos e verificáveis.
2. Não implemente funcionalidades fora do escopo sem justificar.
3. Prefira soluções simples, modulares e fáceis de explicar.
4. Antes de alterar código, explique brevemente o objetivo e os arquivos envolvidos.
5. Após cada implementação, execute os testes relevantes e informe o resultado.
6. Não declare algo como concluído sem evidência de execução ou teste.
7. Mantenha o README e a documentação atualizados.
8. Não invente métricas, resultados de treinamento ou evidências de avaliação.
9. Não substitua o fine-tuning obrigatório por apenas prompting ou RAG.
10. Não substitua os fluxos LangGraph por uma sequência de chamadas sem grafo explícito.

## Segurança

- Utilizar apenas dados sintéticos.
- Nunca incluir chaves ou segredos no repositório.
- Não produzir prescrições clínicas autônomas.
- Exigir validação humana para recomendações clínicas.
- Identificar fontes e incertezas.
- Registrar eventos de auditoria sem expor desnecessariamente informações sensíveis.

## Qualidade técnica

- Python com tipagem quando útil.
- Funções e módulos com responsabilidades claras.
- Testes com pytest.
- Dependências e versões documentadas.
- Tratamento explícito de erros.
- Separação entre dados, RAG, fine-tuning, fluxos e interface.

## Forma de colaboração

O desenvolvedor está aprendendo IA aplicada.

Ao propor uma implementação, explique:
- Qual problema resolve.
- Por que essa abordagem foi escolhida.
- Como verificar se funciona.
- Quais limitações ou riscos existem.

Evite complexidade desnecessária e alterações grandes de uma só vez.