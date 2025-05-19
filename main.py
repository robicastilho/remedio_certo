# Instalação dos pacotes necessários (para uso local ou GCP)
# Removido ngrok
#!pip -q install flask requests python-magic google-genai google-adk

from flask import Flask, request, jsonify
import requests
import magic
import json
import os
from datetime import date, datetime, timedelta
from threading import Thread
import time
from google import genai
from google.genai import types
from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import google_search


# 🔒 Validação de variáveis de ambiente obrigatórias
REQUIRED_ENV_VARS = [
    "WHATS_VERIFY_TOKEN",
    "WHATS_ACCESS_TOKEN",
    "WHATS_PHONE_NUMBER_ID",
    "GOOGLE_API_KEY"
]

missing_vars = [var for var in REQUIRED_ENV_VARS if not os.getenv(var)]
if missing_vars:
    print(f"❌ ERRO: Variáveis de ambiente ausentes: {', '.join(missing_vars)}", file=sys.stderr)
    sys.exit(1)


app = Flask(__name__)

# Carregue variáveis de ambiente no deploy no GCP
VERIFY_TOKEN = os.getenv('WHATS_VERIFY_TOKEN')
ACCESS_TOKEN = os.getenv('WHATS_ACCESS_TOKEN')
PHONE_NUMBER_ID = os.getenv('WHATS_PHONE_NUMBER_ID')
os.environ['GOOGLE_API_KEY'] = os.getenv('GOOGLE_API_KEY')
model_name = "gemini-2.5-flash-preview-04-17"

user_contexts = {}
SESSION_TIMEOUT_MINUTES = 30

# Verifica sessões inativas a cada 5 minutos
# (mantém o mesmo)
def session_cleanup_loop():
    while True:
        now = datetime.utcnow()
        to_remove = []
        for number, context in user_contexts.items():
            if context.get("imagem_bytes") or context.get("especialidade") or context.get("sintomas"):
                context_time = context.get("last_updated")
                if not context_time:
                    continue
                if now - context_time > timedelta(minutes=SESSION_TIMEOUT_MINUTES):
                    send_message(number, "Sua sessão expirou por inatividade. Por favor, envie a receita ou informações novamente para recomeçar.")
                    to_remove.append(number)
        for number in to_remove:
            user_contexts.pop(number)
        time.sleep(300)

Thread(target=session_cleanup_loop, daemon=True).start()

def call_agent(agent: Agent, prompt: str) -> str:
    session_service = InMemorySessionService()
    session = session_service.create_session(app_name=agent.name, user_id="user1", session_id="session1")
    runner = Runner(agent=agent, app_name=agent.name, session_service=session_service)
    content = types.Content(role="user", parts=[types.Part(text=prompt)])
    final_response = ""
    for event in runner.run(user_id="user1", session_id="session1", new_message=content):
        if event.is_final_response():
            for part in event.content.parts:
                if part.text is not None:
                    final_response += part.text + "\n"
    return final_response.strip()

def call_agent_multimodal_com_bytes(agent: Agent, message_text: str, image_bytes: bytes, mime_type: str) -> str:
    session_service = InMemorySessionService()
    session = session_service.create_session(app_name=agent.name, user_id="user1", session_id="session1")
    runner = Runner(agent=agent, app_name=agent.name, session_service=session_service)

    list_of_parts = [types.Part(text=message_text)]
    if image_bytes:
        list_of_parts.append(types.Part(inline_data=types.Blob(mime_type=mime_type, data=image_bytes)))

    content = types.Content(role="user", parts=list_of_parts)
    final_response = ""
    for event in runner.run(user_id="user1", session_id="session1", new_message=content):
        if event.is_final_response():
            for part in event.content.parts:
                if part.text is not None:
                    final_response += part.text + "\n"
    return final_response.strip()

def agente_identificador_especialidade(texto_para_analise: str) -> str:
    instrucao_para_agente = """
    Você é um assistente inteligente especializado em terminologia e reconhecimento de especialidades médicas.
    Sua principal tarefa é analisar o texto fornecido pelo usuário e identificar se ele se refere a uma especialidade médica conhecida.

    Siga estas diretrizes rigorosamente:

    1.  **Análise do Texto:** Examine o texto de entrada cuidadosamente. O texto pode conter o nome de uma especialidade (ex: "Oftalmologia"), o nome de um médico especialista (ex: "urologista") ou uma descrição da área (ex: "especialista em doenças de pele").

    2.  **Identificação da Especialidade:**
        * Se você identificar claramente uma especialidade médica, retorne o nome padronizado dessa especialidade.
            Exemplos de mapeamento para o retorno esperado:
            - Se a entrada for "cardiologia" ou "médico do coração" ou "cardiologista", retorne "Cardiologista".
            - Se a entrada for "dermatologia" ou "especialista de pele" ou "dermatologista", retorne "Dermatologista".
            - Se a entrada for "urologia" ou "urologista", retorne "Urologista".
            - Se a entrada for "clínica geral", "médico de família" ou "clinico geral", retorne "clinico geral".
            - Se a entrada for "pediatria" ou "médico de crianças", retorne "Pediatra".
        * Seja flexível com variações comuns, mas mantenha a precisão.

    3.  **Retorno Padrão (Fallback):**
        * Se o texto fornecido NÃO contiver uma referência clara a uma especialidade médica reconhecível,
        * OU se for ambíguo,
        * OU se for um texto genérico que não se relaciona com especialidades médicas,
        * ENTÃO, você DEVE retornar a string "clinico geral".

    4.  **Formato da Saída:**
        * Sua resposta deve ser ÚNICA E EXCLUSIVAMENTE o nome da especialidade identificada (conforme item 2) ou a string "clinico geral" (conforme item 3).
        * Não inclua NENHUMA palavra, frase, explicação, pontuação ou formatação adicional. Apenas a string resultante.
          Por exemplo, se identificar "Urologista", sua saída deve ser exatamente "Urologista". Se o fallback for ativado, sua saída deve ser exatamente "None".
    """
    identificador_agente = Agent(
        name="agente_identificador_especialidade",
        model=model_name,
        instruction=instrucao_para_agente,
        description="Agente especialista em identificar especialidades médicas em texto ou retornar 'clinico geral'."
    )
    entrada_para_agente = f"Por favor, analise o seguinte texto para identificar a especialidade médica: \"{texto_para_analise}\""
    saida = call_agent(identificador_agente, entrada_para_agente).strip()
    print(f"Especialidade: {saida}")
    return saida if saida != "None" else None;

def agente_identificador_sintomas(sintomas_paciente: str) -> str:
    instrucao_para_agente = """
    Você é um assistente médico altamente competente, treinado para identificar e extrair sintomas ou doenças de pacientes
    a partir de descrições textuais. Sua principal tarefa é analisar o texto fornecido pelo usuário
    e determinar de forma concisa quais são os sintomas ou doenças relatados.

    Siga estas diretrizes rigorosamente:

    1.  **Análise do Texto:** Examine o texto de entrada fornecido pelo usuário com atenção. O texto pode conter
        descrições de dores, desconfortos, alterações no estado de saúde ou outras condições que
        configuram sintomas.

    2.  **Identificação e Extração de Sintomas ou Doenças:**
        * Se o texto descrever um ou mais sintomas ou doenças de forma clara (por exemplo, "Estou com uma dor de cabeça terrível",
            "Sinto febre e calafrios constantes", "Tenho tido muita tosse seca e uma leve dor de garganta", "Estou com covid"),
            seu objetivo é extrair a essência desses sintomas ou doenças.
        * Seja conciso e direto na descrição do sintoma ou doença retornado.

    3.  **Formato do Retorno (Sintomas ou doenças Encontrados):**
        * Se um único sintoma ou doença for identificado, retorne APENAS a descrição desse sintoma ou doença.
        * Se múltiplos sintomas ou doenças forem identificados, liste-os em uma única string, de forma natural.

    4.  **Retorno Padrão (Sintomas ou doenças Não Identificados/Desconhecidos):**
        * Se o texto fornecido NÃO descrever nenhum sintoma ou doença médico reconhecível,
        * OU se a descrição for excessivamente vaga, subjetiva demais e não permitir a identificação de um sintoma ou doença específico,
        * ENTÃO, você DEVE retornar a string "são desconhecidos".

    5.  **Exclusividade da Saída:**
        * Sua resposta deve ser ÚNICA E EXCLUSIVAMENTE a string contendo os sintomas ou doenças identificados (conforme item 3)
            ou a string "None" (conforme item 4).
        * Não inclua NENHUMA palavra, frase introdutória, explicação,
            pontuação desnecessária ou formatação extra. Apenas a string resultante.
    """
    identificador_sintomas_agente = Agent(
        name="agente_identificador_sintomas",
        model=model_name,
        instruction=instrucao_para_agente,
        description="Agente especialista em identificar sintomas de pacientes em texto ou retornar 'são desconhecidos'."
    )
    entrada_para_agente = f"Por favor, analise a descrição a seguir e identifique os sintomas do paciente: \"{sintomas_paciente}\""
    saida = call_agent(identificador_sintomas_agente, entrada_para_agente).strip()
    print(f"Sintomas: {saida}")
    return saida if saida != "None" else None;

def agente_farmaceutico(image_bytes: bytes, mime_type: str, especialidade_medica: str, sintomas_paciente: str) -> str:
    farmaceutico = Agent(
        name="agente_farmaceutico",
        model=model_name,
        instruction=f"""
        Você é um Farmacêutico altamente experiente e especializado em decifrar receitas médicas manuscritas, com profundo conhecimento em farmacologia, caligrafia médica e abreviações comuns.

        **Você já tem a informação de que a especialidade médica do prescritor é {especialidade_medica} e os sintomas do paciente {sintomas_paciente}.**

        Sua tarefa principal é analisar uma imagem de uma receita médica manuscrita e extrair as informações cruciais sobre os medicamentos prescritos.

        Para cada item identificado, envie o resultado formatado com clareza em estilo Markdown para leitura via WhatsApp. Use a estrutura:

        📌 *Medicamento:* Nome do medicamento
        💊 *Posologia:* Dose, via, frequência
        ⏳ *Duração:* Tempo de tratamento
        📈 *Confiança:* ALTO / MÉDIO / BAIXO
        📝 *Observações:* Dúvidas ou alternativas identificadas

        Separe cada medicamento com uma linha de separação:

        ---

        Evite repetições. Use poucos emojis. Foque em clareza e objetividade. Formate com asteriscos (*) os campos para facilitar leitura no WhatsApp.

        **Passos para a identificação (em ordem de prioridade):**

        1.  **Decifração Direta:** Comece lendo e transcrevendo o que o médico escreveu com a maior precisão possível.
        2.  **Análise de Contexto e Caligrafia:** Utilize seu conhecimento em farmacologia, abreviações comuns em receitas e, crucialmente, a **informação da especialidade médica e sintomas do paciente** para inferir palavras ou partes ilegíveis.
        3.  **Busca por Similares (Grafia):** Se a palavra for incerta ou o medicamento encontrado não seja para tratamento dos sintomas informados, pesquise por nomes de medicamentos ou termos de posologia com grafia similar ou fonética próxima que se encaixem no contexto.
        4.  **Identificação de Ilegíveis:** Se, após os passos anteriores, uma parte da receita permanecer impossível de decifrar, classifique-a como "Não identificado" ou "Ilegível" e atribua um grau de confiança "BAIXO". **Nunca invente informações ou nomes de medicamentos.**
        5.  **Validação e Refinamento Contextual:**
            * Com base no nome do medicamento que você identificou (mesmo que com confiança baixa), **pesquise no (google_search) por medicamentos que tenham nomes iguais ou muito similares**.
            * Simultaneamente, avalie se os medicamentos encontrados nessa pesquisa e o medicamento originalmente identificado **possuem ação terapêutica compatível com os sintomas informados pelo paciente e são comumente prescritos por médicos com a mesma especialidade** para essa condição.
            * Se esta análise contextual sugerir fortemente um medicamento diferente do inicialmente decifrado (mas graficamente similar e contextualmente mais plausível), **corrija a informação do medicamento** e ajuste o Grau de Confiança para cima, justificando a alteração em "Observações/Incertezas". Caso a compatibilidade seja baixa, mas a legibilidade da receita persista, mantenha a incerteza e o Grau de Confiança original ou até o diminua.
        """,
        description="Agente farmacêutico que analisa receitas médicas manuscritas e usa o Google Search para validação.",
        tools=[google_search]
    )

    especialidade_medica = especialidade_medica if especialidade_medica != None else "clínico geral"
    sintomas_paciente = sintomas_paciente if sintomas_paciente != None else "são desconhecidos"

    entrada = (
        f"Especialidade médica do prescritor: {especialidade_medica}\n"
        f"Sintomas do paciente: {sintomas_paciente}\n\n"
        "Por favor, analise a IMAGEM da receita médica que acompanha esta solicitação."
    )
    return call_agent_multimodal_com_bytes(farmaceutico, entrada, image_bytes, mime_type)

def agente_buscador_medicamentos_online(receita_analisada: str, data_de_hoje: str) -> str:
    instrucao_para_agente = f"""
    Você é um assistente de pesquisa especializado em encontrar opções de compra online de medicamentos no Brasil.
    Sua data de conhecimento atual é {data_de_hoje}.

    Abaixo está o conteúdo de uma receita analisada:
    "{receita_analisada}"

    Sua tarefa é analisar o texto de uma receita fornecida, identificar cada medicamento e, para cada um,
    usar a ferramenta de busca do Google (google_search) para encontrar até três opções de compra online.

    📦 *Medicamento:* [Nome do Medicamento Conforme Identificado/Pesquisado]
    🏪 *Farmácia:* [Nome do Site/Farmácia]
    💰 *Preço:* [Preço Encontrado ou "Preço não encontrado"]
    🔗 *Link:* URL curta e clicável

    Liste no máximo 3 opções por medicamento, separadas por "---".
    Se não encontrar opções, diga:
    _Nenhuma opção de compra online encontrada para este medicamento._

    Formate cada medicamento com um cabeçalho tipo:

    ➕ *Opções para [NOME DO MEDICAMENTO]*

    Use emojis moderadamente e garanta boa leitura no WhatsApp.

    **Considerações Adicionais:**
        * Sempre utilize a ferramenta de busca do Google (google_search) para cada medicamento. Não use conhecimento prévio sobre preços ou URLs,
            pois eles mudam constantemente.
        * Seja objetivo e forneça apenas as informações solicitadas no formato especificado.
        * Certifique-se de que as URLs sejam válidas e apontem para sites brasileiros.
    """

    buscador_medicamentos_agente = Agent(
        name="agente_buscador_medicamentos_online",
        model=model_name,
        instruction=instrucao_para_agente,
        description="Agente que busca informações de compra online para medicamentos usando o Google Search.",
        tools=[google_search]
    )
    entrada_para_agente = "Por favor, processe a receita fornecida e encontre as opções de compra."
    return call_agent(buscador_medicamentos_agente, entrada_para_agente).strip()

def enviar_typing_indicator(message_id: str):
    url = f"https://graph.facebook.com/v22.0/{PHONE_NUMBER_ID}/messages"
    headers = {
        "Authorization": f"Bearer {ACCESS_TOKEN}",
        "Content-Type": "application/json"
    }
    payload = {
        "messaging_product": "whatsapp",
        "status": "read",
        "message_id": message_id,
        "typing_indicator": {
            "type": "text"
        }
    }
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code != 200:
        print(f"Erro ao enviar indicador de digitação: {response.status_code} - {response.text}")

@app.route("/webhook", methods=["GET"])
def verify():
    print(" Verificação GET Recebida")
    mode = request.args.get("hub.mode")
    token = request.args.get("hub.verify_token")
    challenge = request.args.get("hub.challenge")
    if mode == "subscribe" and token == VERIFY_TOKEN:
        return challenge, 200
    return "Token inválido", 403


@app.route("/webhook", methods=["POST"])
def webhook():

    print("🔔 POST recebido no /webhook")
    
    #data = request.get_json()
    #print("Mensagem recebida:", json.dumps(data, indent=2))

    try:

        data = request.get_json()
        print("📨 Conteúdo recebido:", json.dumps(data, indent=2))
        
        entry = data["entry"][0]
        changes = entry["changes"][0]
        value = changes["value"]
        messages = value.get("messages")

        if not messages:
            return jsonify(status="sem mensagem"), 200

        msg = messages[0]
        message_id = msg.get("id")
        from_number = msg["from"]
        msg_type = msg["type"]

        context = user_contexts.get(from_number)

        if not context:
            context = {
                "especialidade": None,
                "sintomas": None,
                "imagem_bytes": None,
                "mime_type": None,
                "last_updated": datetime.utcnow(),
                "last_message_id": None
            }

        if context.get("last_message_id") == message_id:
            return jsonify(status="mensagem duplicada ignorada"), 200

        context["last_message_id"] = message_id

        if msg_type == "image":
            media_id = msg["image"]["id"]
            media_url_resp = requests.get(
                f"https://graph.facebook.com/v22.0/{media_id}",
                headers={"Authorization": f"Bearer {ACCESS_TOKEN}"}
            )
            media_url = media_url_resp.json().get("url")
            image_resp = requests.get(media_url, headers={"Authorization": f"Bearer {ACCESS_TOKEN}"})
            context["imagem_bytes"] = image_resp.content
            context["mime_type"] = magic.from_buffer(context["imagem_bytes"], mime=True)

        elif msg_type == "text":
            texto = msg["text"]["body"].lower().strip()
            if not context["especialidade"]:
                context["especialidade"] = agente_identificador_especialidade(texto)
            elif not context["sintomas"]:
                context["sintomas"] = agente_identificador_sintomas(texto)

        context["last_updated"] = datetime.utcnow()
        user_contexts[from_number] = context

        if all([context["imagem_bytes"], context["mime_type"], context["especialidade"], context["sintomas"]]):
            send_message(from_number, "Espere um pouquinho que agora a IA vai fazer a mágica...")
            enviar_typing_indicator(message_id)

            resultado = agente_farmaceutico(
                image_bytes=context["imagem_bytes"],
                mime_type=context["mime_type"],
                especialidade_medica=context["especialidade"],
                sintomas_paciente=context["sintomas"]
            )
            send_message(from_number, resultado)

            send_message(from_number, "Agora deixa eu dar um Google aqui e ver os preços pra você. Espera só um pouquinho")
            enviar_typing_indicator(message_id)
            precos = agente_buscador_medicamentos_online(resultado, str(date.today()))
            send_message(from_number, precos)

            user_contexts.pop(from_number)
        else:
            if not context["imagem_bytes"]:
                send_message(from_number, "Por favor, envie uma foto da receita médica.")
            elif not context["especialidade"]:
                send_message(from_number, "Qual é a especialidade médica do prescritor?")
            elif not context["sintomas"]:
                send_message(from_number, "Quais são os sintomas ou doenças do paciente?")

    except Exception as e:
        import traceback
        print("❌ Erro no webhook POST:", e)
        traceback.print_exc()  # Isso garante que o erro vá para os logs do Cloud Run
        return jsonify({"error": "erro interno no servidor"}), 500

    return jsonify(status="ok"), 200

@app.route("/favicon.ico")
def favicon():
    return '', 204




def send_message(to_number, message_text):
    url = f"https://graph.facebook.com/v22.0/{PHONE_NUMBER_ID}/messages"
    headers = {
        "Authorization": f"Bearer {ACCESS_TOKEN}",
        "Content-Type": "application/json"
    }
    payload = {
        "messaging_product": "whatsapp",
        "to": to_number,
        "type": "text",
        "text": {"body": message_text[:4096]}
    }
    response = requests.post(url, headers=headers, json=payload)
    print("Resposta enviada:", response.status_code, response.text)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 8080)))
