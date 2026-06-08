import asyncio


async def analyze_event(event_id, client, urls):
    """Fan out to channel services in parallel, then fuse. Returns a FusionResult dict."""
    payload = {"event_id": event_id}

    async def call(channel):
        try:
            resp = await client.post(urls[channel], json=payload)
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            return {"event_id": event_id, "channel": channel, "score": 0.0,
                    "label": "neutral", "raw": {}, "ok": False, "error": str(exc)}

    channels = await asyncio.gather(call("nlp"), call("audio"), call("vision"))

    fuse_resp = await client.post(urls["fusion"],
                                  json={"event_id": event_id, "channels": list(channels)})
    return fuse_resp.json()
