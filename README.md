# Implementação de um método para determinação de máximos e mínimos dos espectros de XANES

Emelyn Alves <sup>1</sup>, James Moraes de Almeida <sup>1</sup> e Santiago José Alejandro Figueroa <sup>2,3</sup>

1: Ilum, Escola de Ciência. 
2: Centro Nacional de Pesquisa em Energia e Materiais (CNPEM), Laboratório Nacional de Luz Síncrotron (LNLS), Grupo QUATI. 
3: Instituto de Química, Unicamp. 

Este trabalho é parte do projeto "MÉTODOS DE APRENDIZADO DE MÁQUINA A PARTIR DE ESPECTROS DE RAIOS X USANDO SIMULAÇÕES E DESCRIPTORES", desenvolvido por [Cauê Gomes Correia dos Santos](https://github.com/CaueSantos1812) e Emelyn Alves durante a disciplina de Iniciação à Pesquisa III do 4º semestre do curso de Bacharelado em Ciência e Tecnologia da Ilum, Escola de Ciência. Para mais informações sobre o projeto, baixe o arquivo relatorio, neste github. 

Este trabalho visa implementar o Método Rising-Sun, descrito no artigo [“The Rising Sun Envelope Method: an automatic and accurate peak location technique for XANES measurements”](https://pubs.acs.org/doi/epdf/10.1021/acs.jpca.9b11712?ref=article_openPDF), para a identificação automática e precisa de máximos e mínimos nos espectros XANES, utilizando linguagem de programação Python e arquivos '.xdi' da biblioteca [Cruzeiro do Sul Utils](https://github.com/jamesmalmeida/Cruzeiro-do-Sul-Utils). A implementação pode facilitar a comparação de espectros, aprimorando a análise de dados na linha de luz QUATI (QUick x-ray Absorption spectroscopy for TIme and space-resolved experiments) e possibilitando futuras aplicações com aprendizado de máquina.

## Status 
Implementação realizada. 

## ⚠️ Avisos!
*O codigo passará por refinos de organização e limpeza.* Quaisquer mudanças no código serão atualizadas neste github. Alguns espectros não apresentam todos os seus pontos identificados, ajustes nos parâmetros limiares devem ser realizados para melhor identificação dos pontos, de acordo com [Rafael Monteiro](https://github.com/rafael-a-monteiro-math), um dos desenvolvedores do Método Rising Sun. A análise sequencial de espectros é custosa computacionalmente, logo, é recomendado o uso de um HPC para análise de mais de 20 espectros. 

## Observação
O código base do Método Rising Sun pode ser adquirido a partir do seguinte repositório: [Rising_Sun_Envelope_Method](https://github.com/rafael-a-monteiro-math/Rising_Sun_Envelope_Method). 

## Requisitos 
- JupyterLab - ambiente de execução do código;
- Python - linguagem de programação utilizada no script; 
- Arquivos .xdi - formato de arquivo processado pelo código;

##  Arquivos
- TESTE.py: script em Pyhton da implementação do Método Rising Sun, nele há todas as funções necessárias para a execução do código;
- RUN.ipynb: notebook em Jupyter que executa o script;
- CZDS_Utils: pasta de arquivos .py para biblioteca Cruzeiro do Sul Utils; 
- teste_xdi: pasta com arquivos .xdi para testes de identificação de pontos _(Adicione a essa pasta os arquivos .xdi que serão processados)_;
- relatorio: texto do relatório final apresentado na disciplina de Iniciação à Pesquisa III _(Verifique esse documento para mais informações sobre o processo de implementação)_

## Passos 
1. Baixe os arquivos disponibilizados neste github;
2. Mantenha os arquivos baixados em um mesmo ambiente "x", por exemplo;
3. Crie as seguintes pastas *VAZIAS* no ambiente "x": XANES-csv e XANES-derivatives, essas pastas armazenarão os arquivos .csv e os valores de primeira e segunda derivada, que são gerados pelo cógido TESTE.py;
4. No JupyterLab abra o notebook RUN.ipynb;
5. Execute o código! É esperado que o código processe o arquivo .xdi e realize a identificação dos pontos a partir da plotagem do gráfico entitulado: XANES spectrum with marked peaks. Observação: Por hora, gráficos e textos anteriores a identificação de pontos são mostrados pelo notebook RUN.ipynb.

## Dúvidas, comentários ou sugestões? :)
Entre em contato: emelyn23017@ilum.cnpem.br, [LinkedIn - Emelyn Alves](https://www.linkedin.com/in/emelyn-alves-5362532a0/), +5519987208765. 
