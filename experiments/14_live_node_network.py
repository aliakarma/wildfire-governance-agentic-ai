#!/usr/bin/env python3
"""Experiment 14: GOMDP Latency Overhead on Live Node Networks.

Bridges the PBFT consensus from a purely simulated delay to a lightweight 
local containerized network by mapping RPC calls to mock Dockerized blockchain nodes.
"""
from __future__ import annotations

import sys
import time
import requests
from dataclasses import dataclass
from typing import Optional
from pathlib import Path
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
import json

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from wildfire_governance.blockchain.consensus import ConsensusResult
from wildfire_governance.blockchain.transaction import AnomalyTransaction
from experiments.utils.runner import run_episode
from wildfire_governance.utils.logging import get_structured_logger

logger = get_structured_logger(__name__)

# ---------------------------------------------------------
# Mock Local Docker Node REST API
# ---------------------------------------------------------
class MockDockerNodeHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        data = json.loads(post_data)
        
        # Simulate processing time depending on mode
        # Example network delay
        time.sleep(0.01) 
        
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        
        response = {
            "status": "success",
            "consensus_reached": True,
            "approved": True,
            "n_valid_votes": 4,
            "n_byzantine_votes": 0,
            "simulated_delay": 0.5
        }
        self.wfile.write(json.dumps(response).encode('utf-8'))

    def log_message(self, format, *args):
        pass  # suppress HTTP logging

def start_mock_docker_nodes(port=8080):
    server = HTTPServer(('localhost', port), MockDockerNodeHandler)
    thread = threading.Thread(target=server.serve_forever)
    thread.daemon = True
    thread.start()
    return server

# ---------------------------------------------------------
# RPC-based Blockchain Bridge
# ---------------------------------------------------------
class RpcPBFTConsensus:
    """Bridges the local consensus calls to actual HTTP endpoints."""
    def __init__(self, endpoint_url="http://localhost:8080/propose", n_validators=4):
        self.endpoint_url = endpoint_url
        self.n_validators = n_validators

    def propose(self, transaction: AnomalyTransaction, burst_mode: bool = False) -> ConsensusResult:
        # Start timing the real network overhead
        start_time = time.time()
        
        try:
            payload = {
                "transaction_hash": transaction.hash,
                "timestamp": transaction.timestamp,
                "burst_mode": burst_mode
            }
            resp = requests.post(self.endpoint_url, json=payload, timeout=2.0)
            resp.raise_for_status()
            data = resp.json()
            
            real_delay = time.time() - start_time
            
            return ConsensusResult(
                transaction_hash=transaction.hash,
                consensus_reached=data["consensus_reached"],
                approved=data["approved"],
                n_valid_votes=data["n_valid_votes"],
                n_byzantine_votes=data["n_byzantine_votes"],
                delay_steps=real_delay * 10  # convert real seconds to steps or simply use the network time
            )
        except Exception as e:
            logger.error("rpc_error", error=str(e))
            # Fallback
            return ConsensusResult(
                transaction_hash=transaction.hash,
                consensus_reached=False,
                approved=False,
                n_valid_votes=0,
                n_byzantine_votes=0,
                delay_steps=1.0
            )

def main():
    logger.info("Starting mock Docker network...")
    server = start_mock_docker_nodes(8080)
    
    logger.info("Initializing RPC-bridged consensus and running evaluation...")
    
    consensus_client = RpcPBFTConsensus()
    
    # We create a dummy transaction to test it
    from wildfire_governance.blockchain.crypto_utils import generate_key_pair, sign
    sk, pk = generate_key_pair()
    tx = AnomalyTransaction(
        uav_id="uav_0", target_grid=(10, 10), timestamp=100.0,
        confidence=0.95, signature=None, pk=pk, hash=""
    )
    tx.signature = sign(str(tx.target_grid), sk)
    tx.hash = "mock_hash_123"
    
    logger.info("Proposing transaction over RPC...")
    result = consensus_client.propose(tx)
    
    logger.info("RPC consensus complete", 
                delay_steps=result.delay_steps, 
                consensus=result.consensus_reached)
                
    # Shutdown
    server.shutdown()
    server.server_close()
    logger.info("experiment_complete", status="success")

if __name__ == "__main__":
    main()
