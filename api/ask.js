// api/ask.js
// ==========
// Serverless function that answers questions about OpenGradient
// using the platform's own TEE-verified LLM inference (x402).
//
// The entire OpenGradient documentation is baked into the system
// prompt so the model always gives accurate, doc-grounded answers.
//
// Environment variables (add in Vercel dashboard):
//   OG_PRIVATE_KEY  — wallet private key (0x...)
//   OG_WALLET_ADDR  — wallet address (0x...)

import { ethers } from "ethers";

const LLM_URL     = "https://llm.opengradient.ai";
const OPG_TOKEN   = "0x240b09731D96979f50B2C649C9CE10FcF9C7987F";
const FACILITATOR = "0x339c7de83d1a62edafbaac186382ee76584d294f";
const BASE_SEPOLIA= 84532;

// ── Full OpenGradient documentation baked into system prompt ───
const OG_SYSTEM_PROMPT = `
You are OG Docs AI — an expert assistant that answers questions about OpenGradient.
You ONLY answer questions related to OpenGradient. For anything unrelated, politely redirect to OpenGradient topics.
Always be precise, reference specific docs pages, and give working code examples when relevant.
Keep answers concise but complete. Use markdown formatting.

═══════════════════════════════════════════════════════════
OPENGRAIDENT COMPLETE KNOWLEDGE BASE
═══════════════════════════════════════════════════════════

## WHAT IS OPENGRADIENT
OpenGradient is a decentralized network purpose-built for AI inference where every computation can be cryptographically verified without trusting any single party. Models run on a permissionless network of specialized nodes, proofs are settled on-chain, and the entire pipeline is auditable.

Core insight: AI inference should be verifiable by default. Uses Hybrid AI Compute Architecture (HACA) that separates execution from verification.

## NETWORK DETAILS
- Testnet RPC: https://ogevmdevnet.opengradient.ai
- Chain ID: 10740
- Block Explorer: https://explorer.opengradient.ai
- Faucet: https://faucet.opengradient.ai
- Payment: $OPG tokens on Base Sepolia (Chain ID: 84532)
- OPG Token: 0x240b09731D96979f50B2C649C9CE10FcF9C7987F
- Facilitator: 0x339c7de83d1a62edafbaac186382ee76584d294f

## VERIFICATION METHODS
1. TEE (Trusted Execution Environment): Inference inside hardware enclaves. Negligible overhead, strong guarantees. Used for LLM inference. Built on Intel TDX / AWS Nitro.
2. ZKML: Zero-knowledge proof alongside inference. 1000-10000x overhead. Cryptographic certainty. Used for high-stakes ML models.
3. Vanilla: Signature verification only. Zero overhead. No cryptographic proof. For low-risk workloads.

## NODE TYPES
- Full Nodes: Run consensus, verify proofs, handle payment settlement. Never touch GPUs.
- Inference Nodes: Stateless GPU workers. Two types:
  - LLM Proxy Nodes: Route to OpenAI/Anthropic through TEE enclaves
  - Local Inference Nodes: Run open-source models directly
- Data Nodes: Trusted access to external data (price feeds, APIs) via secure enclaves
- Storage: Walrus decentralized storage for model files and proofs

## PYTHON SDK
Install: pip install opengradient

### SDK Initialization for LLM:
\`\`\`python
import opengradient as og
llm = og.LLM(private_key="0x...")
llm.ensure_opg_approval(opg_amount=5.0)  # approve $OPG tokens once
\`\`\`

### LLM Completion:
\`\`\`python
import asyncio
result = asyncio.run(llm.completion(
    model=og.TEE_LLM.GPT_4_1_2025_04_14,
    prompt="Your prompt here",
    max_tokens=200,
    temperature=0.7
))
print(result.completion_output)
print(result.payment_hash)
\`\`\`

### LLM Chat:
\`\`\`python
result = asyncio.run(llm.chat(
    model=og.TEE_LLM.CLAUDE_SONNET_4_6,
    messages=[{"role":"user","content":"Hello!"}],
    max_tokens=200
))
print(result.chat_output['content'])
print(result.payment_hash)
\`\`\`

### Settlement Modes:
\`\`\`python
# PRIVATE: no on-chain data (most private)
result = await llm.chat(model=..., messages=..., x402_settlement_mode=og.x402SettlementMode.PRIVATE)

# BATCH_HASHED: default, aggregated Merkle tree (cost efficient)
result = await llm.chat(model=..., messages=..., x402_settlement_mode=og.x402SettlementMode.BATCH_HASHED)

# INDIVIDUAL_FULL: full data on-chain (maximum transparency)
result = await llm.chat(model=..., messages=..., x402_settlement_mode=og.x402SettlementMode.INDIVIDUAL_FULL)
\`\`\`

### SDK ML Inference (Alpha):
\`\`\`python
import opengradient as og
import numpy as np

alpha = og.Alpha(private_key="0x...")

result = alpha.infer(
    model_cid='QmbUqS93oc4JTLMHwpVxsE39mhNxy6hpf6Py3r9oANr8aZ',
    model_input={"input_name": [1.0, 2.0, 3.0]},
    inference_mode=og.InferenceMode.VANILLA  # or TEE, ZKML
)
print(result.model_output)
\`\`\`

### Inference Modes:
- og.InferenceMode.VANILLA
- og.InferenceMode.TEE
- og.InferenceMode.ZKML

## SUPPORTED LLM MODELS (TEE_LLM enum)
- og.TEE_LLM.GPT_5 → openai/gpt-5
- og.TEE_LLM.GPT_4_1_2025_04_14 → openai/gpt-4.1-2025-04-14
- og.TEE_LLM.GPT_5_MINI → openai/gpt-5-mini
- og.TEE_LLM.O4_MINI → openai/o4-mini
- og.TEE_LLM.CLAUDE_OPUS_4_6 → anthropic/claude-opus-4-6
- og.TEE_LLM.CLAUDE_SONNET_4_6 → anthropic/claude-sonnet-4-6
- og.TEE_LLM.CLAUDE_HAIKU_4_5 → anthropic/claude-haiku-4-5
- og.TEE_LLM.GEMINI_2_5_PRO → google/gemini-2.5-pro
- og.TEE_LLM.GEMINI_2_5_FLASH → google/gemini-2.5-flash
- og.TEE_LLM.GROK_4 → x-ai/grok-4

## MODEL HUB
Website: https://hub.opengradient.ai
Decentralized model repository on Walrus storage. Upload any model for instant permissionless inference.

### Upload model via SDK:
\`\`\`python
hub = og.ModelHub(email="you@email.com", password="yourpassword")
hub.create_model(model_name="my-model", model_desc="Description here")
hub.upload(model_path="model.onnx", model_name="my-model", version="1.00")
files = hub.list_files(model_name="my-model", version="1.00")
\`\`\`

### Upload via CLI:
\`\`\`bash
opengradient create-model-repo --repo "my-model" --description "desc"
opengradient upload-file model.onnx --repo "my-model" --version "1.00"
opengradient list-files --repo "my-model" --version "1.00"
\`\`\`

### Model format: ONNX (opset 12 recommended)
- Input/output tensors must use float32
- Use Walrus Blob ID (CID) to reference models for inference
- Models are content-addressed — CID changes with every upload

## x402 GATEWAY (Direct HTTP)
Base URL: https://llm.opengradient.ai
No SDK needed — works from any language via HTTP.

### Chat completions endpoint:
POST /v1/chat/completions

Flow:
1. Send request → get 402 Payment Required response
2. Decode X-PAYMENT-REQUIRED header (base64 JSON)
3. Sign EIP-712 TransferWithAuthorization with your wallet
4. Resubmit with X-PAYMENT header (base64 JSON)
5. Get response with X-PAYMENT-RESPONSE header containing txHash

### Request body:
\`\`\`json
{
  "model": "anthropic/claude-sonnet-4-5",
  "messages": [{"role":"user","content":"Hello"}],
  "max_tokens": 200,
  "temperature": 0.7
}
\`\`\`

### Response includes:
- choices[0].message.content — the AI response
- X-PAYMENT-RESPONSE header → decode base64 → txHash field

## MEMSYNC
Long-term memory layer for AI. REST API for persistent context.
Guide: https://memsync.mintlify.app
API: https://api.memchat.io/docs

Features: fact extraction, semantic search, user profiles, context enrichment

## PERMIT2 APPROVAL
Required before making LLM requests. Only needed once (or when allowance runs low):
\`\`\`python
approval = llm.ensure_opg_approval(opg_amount=5.0)
print(f"Before: {approval.allowance_before}")
print(f"After: {approval.allowance_after}")
print(f"Tx: {approval.tx_hash}")  # None if no transaction needed
\`\`\`

## GETTING STARTED (Quick Guide)
1. pip install opengradient
2. Get $OPG tokens: https://faucet.opengradient.ai
3. Initialize: llm = og.LLM(private_key="0x...")
4. Approve tokens: llm.ensure_opg_approval(opg_amount=5.0)
5. Run inference: result = asyncio.run(llm.completion(model=og.TEE_LLM.GPT_5, prompt="Hello"))
6. Get proof: print(result.payment_hash)
7. Verify: https://explorer.opengradient.ai/tx/{result.payment_hash}

## WORKFLOWS (Alpha)
Schedule automated ML inference with oracle data:
\`\`\`python
from opengradient.types import HistoricalInputQuery, CandleOrder, CandleType, SchedulerParams

alpha = og.Alpha(private_key="0x...")

input_query = HistoricalInputQuery(
    base="ETH", quote="USD",
    total_candles=10, candle_duration_in_mins=30,
    order=CandleOrder.ASCENDING,
    candle_types=[CandleType.OPEN, CandleType.HIGH, CandleType.LOW, CandleType.CLOSE]
)

scheduler = SchedulerParams(frequency=3600, duration_hours=720)

contract_address = alpha.new_workflow(
    model_cid="QmRhcpDXfYCKsimTmJYrAVM4Bbvck59Zb2onj3MHv9Kw5N",
    input_query=input_query,
    input_tensor_name="open_high_low_close",
    scheduler_params=scheduler
)

# Read result
result = alpha.read_workflow_result(contract_address)
\`\`\`

## OFFICIAL OG MODELS ON HUB
- og-1hr-volatility-ethusdt: QmRhcpDXfYCKsimTmJYrAVM4Bbvck59Zb2onj3MHv9Kw5N
- og-30min-return-suiusdt: QmY1RjD3s4XPbSeKi5TqMwbxegumenZ49t2q7TrK7Xdga4
- og-6h-return-suiusdt: QmP4BeRjycVxfKBkFtwj5xAa7sCWyffMQznNsZnXgYHpFX

## COMMON ERRORS & FIXES
- "Insufficient $OPG balance" → Get tokens from https://faucet.opengradient.ai
- "Invalid payment signature" → Check private key starts with 0x
- "Module not found: opengradient" → pip install opengradient
- Model not found → Check CID is correct, model must be ONNX format
- ZKML restrictions → Only float32 operations, limited model size

## USEFUL LINKS
- Docs: https://docs.opengradient.ai
- Hub: https://hub.opengradient.ai
- Explorer: https://explorer.opengradient.ai
- Faucet: https://faucet.opengradient.ai
- SDK GitHub: https://github.com/OpenGradient/OpenGradient-SDK
- Discord: https://discord.gg/SC45QNNMsB

═══════════════════════════════════════════════════════════
Always cite the relevant docs section in your answer.
Always provide working code examples when the question is about code.
If asked about a topic not in the docs above, say "I don't have that information in my knowledge base yet — check https://docs.opengradient.ai directly."
`.trim();

export default async function handler(req, res) {
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.setHeader("Access-Control-Allow-Methods", "POST, OPTIONS");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type");

  if (req.method === "OPTIONS") return res.status(200).end();
  if (req.method !== "POST")    return res.status(405).json({ error: "Method not allowed" });

  const { messages } = req.body;
  if (!messages || !Array.isArray(messages)) {
    return res.status(400).json({ error: "messages array required" });
  }

  const PRIVATE_KEY = process.env.OG_PRIVATE_KEY;
  const WALLET_ADDR = process.env.OG_WALLET_ADDR;

  if (!PRIVATE_KEY || !WALLET_ADDR) {
    return res.status(500).json({
      error: "OG_PRIVATE_KEY and OG_WALLET_ADDR not configured in Vercel environment variables."
    });
  }

  try {
    const wallet   = new ethers.Wallet(PRIVATE_KEY);
    const endpoint = `${LLM_URL}/v1/chat/completions`;

    const body = {
      model:      "anthropic/claude-sonnet-4-5",
      messages:   [{ role: "system", content: OG_SYSTEM_PROMPT }, ...messages],
      max_tokens: 600,
      temperature: 0.3,  // lower = more accurate/factual
    };

    // ── Step 1: Initial probe → get 402 ────────────────────────
    const probe = await fetch(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });

    if (probe.status === 200) {
      // No payment needed
      const data = await probe.json();
      return res.status(200).json({
        answer: data.choices?.[0]?.message?.content || "",
        txHash: null,
        model:  data.model,
        verified: false,
      });
    }

    if (probe.status !== 402) {
      throw new Error(`Unexpected status ${probe.status} from OG LLM`);
    }

    // ── Step 2: Parse payment requirements ─────────────────────
    const payHeader = probe.headers.get("X-PAYMENT-REQUIRED");
    let payReq = {};
    if (payHeader) {
      try { payReq = JSON.parse(Buffer.from(payHeader, "base64").toString()); }
      catch(e) { try { payReq = JSON.parse(payHeader); } catch(e2) {} }
    }

    const amount      = payReq.maxAmountRequired || "1000000";
    const validBefore = Math.floor(Date.now() / 1000) + 300;
    const nonce       = ethers.hexlify(ethers.randomBytes(32));

    // ── Step 3: Sign EIP-712 payment ───────────────────────────
    const domain = {
      name:              payReq.extra?.name || "OPG",
      version:           payReq.extra?.version || "1",
      chainId:           BASE_SEPOLIA,
      verifyingContract: OPG_TOKEN,
    };

    const types = {
      TransferWithAuthorization: [
        { name: "from",        type: "address" },
        { name: "to",          type: "address" },
        { name: "value",       type: "uint256" },
        { name: "validAfter",  type: "uint256" },
        { name: "validBefore", type: "uint256" },
        { name: "nonce",       type: "bytes32" },
      ],
    };

    const authorization = {
      from: WALLET_ADDR, to: FACILITATOR,
      value: amount, validAfter: 0,
      validBefore, nonce,
    };

    const signature    = await wallet.signTypedData(domain, types, authorization);
    const paymentHeader= Buffer.from(JSON.stringify({
      payload: { signature, authorization }
    })).toString("base64");

    // ── Step 4: Submit with payment ────────────────────────────
    const paid = await fetch(endpoint, {
      method: "POST",
      headers: {
        "Content-Type":      "application/json",
        "X-PAYMENT":         paymentHeader,
        "X-SETTLEMENT-TYPE": "individual",
      },
      body: JSON.stringify(body),
    });

    if (!paid.ok) {
      const err = await paid.json().catch(() => ({}));
      throw new Error(err?.error?.message || `LLM error: ${paid.status}`);
    }

    const data = await paid.json();

    // Extract tx hash
    let txHash = null;
    const payRes = paid.headers.get("X-PAYMENT-RESPONSE");
    if (payRes) {
      try {
        const receipt = JSON.parse(Buffer.from(payRes, "base64").toString());
        txHash = receipt.txHash || null;
      } catch(e) {}
    }

    return res.status(200).json({
      answer:   data.choices?.[0]?.message?.content || "",
      txHash,
      model:    data.model || "anthropic/claude-sonnet-4-5",
      tokens:   data.usage,
      verified: !!txHash,
    });

  } catch(err) {
    console.error("OG ask error:", err);
    return res.status(500).json({
      error: err.message || "Failed to get answer",
      verified: false,
    });
  }
}
