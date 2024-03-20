import asyncio

class SharedResource:
    def __init__(self):
        self._data = None
        self._event = asyncio.Event()

    async def set_data(self, data):
        self._data = data
        self._event.set()  # Signal that data is ready

    async def get_data(self):
        await self._event.wait()  # Wait until data is available
        return self._data
