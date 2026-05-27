# Phase 2: Per-Conversation WebSocket + WebRTC Signaling

## Architecture

Two WebSocket roles:
- `/ws/translate` — untouched, handles landmarks/frames for sign language recognition
- `/ws/signal/{conversation_id}` — new, handles WebRTC signaling + translation message relay

## Flow

1. Client processes signs via `/ws/translate` (unchanged flow)
2. After `/ws/translate` returns a `finalize` result, client sends that text as a `translation` message on `/ws/signal/{conversation_id}`
3. Backend stores message in DB + relays to other conversation participants in real-time
4. WebRTC signaling (`offer`, `answer`, `ice_candidate`) relayed through same WS to target participant

## New Files

- `app/services/room_manager.py` — per-conversation room tracking, broadcast, relay
- `app/api/routes/signal.py` — WebSocket endpoint at `/ws/signal/{conversation_id}`

## WebSocket Protocol

### Client → Server

| Type | Payload | Description |
|---|---|---|
| `join` | `{}` | Join conversation room |
| `offer` | `{target_id, sdp}` | WebRTC SDP offer to specific participant |
| `answer` | `{target_id, sdp}` | WebRTC SDP answer to specific participant |
| `ice_candidate` | `{target_id, candidate}` | ICE candidate to specific participant |
| `translation` | `{text, video_url?, audio_url?, confidence_score?}` | Send finalized translation to conversation |
| `ping` | `{}` | Keep-alive |

### Server → Client

| Type | Payload | Description |
|---|---|---|
| `joined` | `{conversation_id, participants}` | Confirmation after join |
| `user_joined` | `{user_id, username}` | Other participant joined |
| `user_left` | `{user_id, username}` | Other participant left |
| `offer` | `{from_id, from_username, sdp}` | Relay from other participant |
| `answer` | `{from_id, sdp}` | Relay from other participant |
| `ice_candidate` | `{from_id, candidate}` | Relay from other participant |
| `translation` | `{from_id, from_username, text, video_url?, audio_url?, confidence_score?, created_at}` | Translation from other participant |
| `pong` | `{}` | Keep-alive response |
| `error` | `{message, code?}` | Error message |

## Auth

JWT token as query param: `/ws/signal/{conversation_id}?token=xxx`
- Validates token
- Checks user is participant of the conversation (via DB lookup)
- Rejects with 4001 if invalid or not participant

## RoomManager

- `rooms: Dict[str, Dict[str, WebSocket]]` — conversation_id → {user_id → WebSocket}
- `join_room(conversation_id, user_id, websocket)` — add to room, broadcast `user_joined`
- `leave_room(conversation_id, user_id)` — remove, broadcast `user_left`
- `relay_to(conversation_id, from_user_id, target_user_id, message)` — send to specific participant
- `broadcast(conversation_id, message, exclude_user_id=None)` — send to all except sender
- Uses `asyncio.Lock` per room for thread safety

## Security

- Participants can only relay WebRTC messages to other participants in the same conversation
- Translation messages are validated and the sender must be a conversation participant
