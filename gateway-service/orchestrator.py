import asyncio


async def analyze_event(event_id, client, urls):
    """Fan out to channel services in parallel, then fuse. Returns a FusionResult dict."""
    payload = {"event_id": event_id}

    async def call(channel):
        resp = await client.post(urls[channel], json=payload)
        return resp.json()

    channels = await asyncio.gather(call("nlp"), call("audio"), call("vision"))

    fuse_resp = await client.post(urls["fusion"],
                                  json={"event_id": event_id, "channels": list(channels)})
    return fuse_resp.json()
