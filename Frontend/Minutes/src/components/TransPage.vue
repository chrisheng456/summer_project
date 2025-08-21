<template>
  <div class="layout">
    <!-- Left: custom TransSidebar with Back -->
    <aside class="sidebar">
      <TransSidebar
        :sections="sideSections"
        :activeIndex="activeIndex"
        @select="handleSelect"
        @back="goBack"
      />
    </aside>

    <!-- Right: main content -->
    <main class="content">
      <div class="topbar">
        <!-- Back button moved to sidebar, only keep search box here -->
        <div class="tools">
          <input
            v-model="keyword"
            class="search"
            type="text"
            placeholder="Search speaker or text..."
          />
        </div>
      </div>

      <h2 class="title">
        Transcription
        <small v-if="detail && detail.name" class="subtitle">· {{ detail.name }}</small>
      </h2>

      <!-- No data -->
      <div v-if="!detail" class="empty">No meeting selected or data not loaded.</div>
      <div v-else-if="!agendasWithLines.length" class="empty">No transcript lines in this meeting.</div>

      <!-- All agendas view -->
      <div v-else-if="activeIndex === -1" class="agenda-list">
        <section v-for="(ag, i) in filteredAgendasAll" :key="ag.id ?? i" class="agenda-card">
          <header class="agenda-header">
            <div class="badge">A{{ i + 1 }}</div>
            <div class="meta">
              <div class="name">{{ formatAgendaTitle(ag) }}</div>
              <div class="count">{{ (ag.lines?.length || 0) }} lines</div>
            </div>
          </header>

          <ul class="line-list">
            <li v-for="(ln, j) in ag._viewLines" :key="j" class="line-item">
              <span class="time" v-if="ln.start !== undefined">{{ toMMSS(ln.start) }}</span>
              <span class="speaker">{{ ln.speaker || 'Unknown' }}:</span>
              <span class="text">{{ ln.text }}</span>
            </li>
          </ul>
        </section>
      </div>

      <!-- Single agenda view (editable) -->
      <div v-else class="agenda-single">
        <section class="agenda-card">
          <header class="agenda-header">
            <div class="badge">A{{ activeIndex + 1 }}</div>
            <div class="meta">
              <div class="name">{{ currentAgendaTitle }}</div>
              <div class="count">{{ currentLines.length }} lines</div>
            </div>

            <!-- Edit actions on the top right -->
            <div class="edit-actions">
              <button v-if="!isEditing" class="btn" @click="startEdit">Edit</button>
              <template v-else>
                <button class="btn primary" @click="saveEdit">Save</button>
                <button class="btn ghost" @click="resetEdit">Reset</button>
              </template>
            </div>
          </header>

          <!-- Time / Summary (editable or read-only) -->
          <div class="edit-fields">
            <div class="field">
              <label>Time:</label>
              <div v-if="isEditing">
                <input v-model="editableAgenda.calculatedStartTime" class="input" />
              </div>
              <div v-else>{{ currentAgenda?.calculatedStartTime || '' }}</div>
            </div>

            <div class="field">
              <label>Summary:</label>
              <div v-if="isEditing">
                <textarea v-model="editableAgenda.summary" class="input textarea" />
              </div>
              <div v-else>{{ currentAgenda?.summary || currentAgenda?.explanation || '' }}</div>
            </div>
          </div>

          <div class="summary-divider"></div>

          <!-- Divider between summary and transcript lines -->
          <ul class="line-list">
            <li v-for="(ln, j) in linesForView" :key="j" class="line-item">
              <span class="time" v-if="ln.start !== undefined">{{ toMMSS(ln.start) }}</span>

              <template v-if="isEditing">
                <input
                  v-model="editableAgenda.lines[j].speaker"
                  class="line-input speaker-input"
                  placeholder="Speaker"
                />
                <textarea
                  v-model="editableAgenda.lines[j].text"
                  class="line-input text-input"
                  placeholder="Text"
                />
              </template>
              <template v-else>
                <span class="speaker">{{ ln.speaker || 'Unknown' }}:</span>
                <span class="text">{{ ln.text }}</span>
              </template>
            </li>
          </ul>
        </section>
      </div>
    </main>
  </div>
</template>

<script setup lang="ts" name="TransPage">
import { computed, ref, type Ref, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { storeToRefs } from 'pinia'

import TransSidebar from '@/components/TransSidebar.vue'
import { useMeetingStore } from '@/stores/meeting'

type TL = { start?: number; end?: number; text: string; speaker: string }
type Agenda = {
  id?: number | string
  number?: string
  title?: string
  calculatedStartTime?: string
  owner?: string
  summary?: string
  explanation?: string
  lines?: TL[]
}
type MeetingDetailMini = { name?: string; agenda?: Agenda[] }

/* Router and store */
const router = useRouter()
const route = useRoute()
const meetingStore = useMeetingStore()
const { details } = storeToRefs(meetingStore) as unknown as {
  details: Ref<Record<string, MeetingDetailMini>>
}

/* Current meeting key and detail */
function asString(v: unknown) {
  if (Array.isArray(v)) return v[0] as string
  if (typeof v === 'string') return v
  return undefined
}
const currentKey = computed(() => {
  let schemeId = asString(route.query.scheme_id)
  let meetingId = asString(route.query.meeting_id)
  if (!schemeId || !meetingId) {
    const k = localStorage.getItem('selectedMeetingKey') || ''
    if (k.includes(':')) {
      const [sid, mid] = k.split(':')
      schemeId = schemeId || sid
      meetingId = meetingId || mid
    }
  }
  return schemeId && meetingId ? `${schemeId}:${meetingId}` : ''
})

const detail = computed<MeetingDetailMini | null>(() => {
  if (!currentKey.value) return null
  return details.value?.[currentKey.value] ?? null
})

/* Sidebar state and filters */
const activeIndex = ref(-1)
const keyword = ref('')
const onlyCurrent = ref(false)

function handleSelect(i: number) {
  activeIndex.value = i
}
function goBack() {
  router.push({ name: 'MeetingNotes' })
}

/* Data derivation */
const agendasWithLines = computed<Agenda[]>(() => {
  const agendas = detail.value?.agenda || []
  return agendas.filter(a => Array.isArray(a?.lines) && a.lines!.length > 0)
})

type SideSection = {
  title: string
  id: string | number
  time: string
  speaker: string
  actions: string[]
  decisions: string[]
  conflicts: string[]
  summary: string
}
const sideSections = computed<SideSection[]>(() => {
  return (agendasWithLines.value || []).map((a, i) => ({
    title: formatAgendaTitle(a),
    id: a.id ?? i,
    time: a.calculatedStartTime ?? '',
    speaker: a.owner ?? '',
    actions: [],
    decisions: [],
    conflicts: [],
    summary: a.summary ?? a.explanation ?? ''
  }))
})

const currentAgenda = computed<Agenda | null>(() => {
  if (activeIndex.value < 0) return null
  return agendasWithLines.value[activeIndex.value] || null
})
const currentAgendaTitle = computed(() => {
  const a = currentAgenda.value
  return a ? formatAgendaTitle(a) : ''
})
const currentLines = computed<TL[]>(() => {
  const a = currentAgenda.value
  return Array.isArray(a?.lines) ? (a!.lines as TL[]) : []
})

/* Search filter (onlyCurrent logic retained but no UI button) */
const filteredAgendasAll = computed(() => {
  const kw = keyword.value.trim().toLowerCase()
  const cur = currentAgenda.value

  return agendasWithLines.value
    .filter(a => (onlyCurrent.value ? a === cur : true))
    .map(a => {
      const lines = (a.lines || []) as TL[]
      const viewLines = !kw
        ? lines
        : lines.filter(
            l =>
              (l.speaker || '').toLowerCase().includes(kw) ||
              (l.text || '').toLowerCase().includes(kw)
          )
      return { ...a, _viewLines: viewLines }
    })
})

/* ======== Editing: time/summary + transcript lines ======== */
const isEditing = ref(false)
const editableAgenda = ref<{ calculatedStartTime: string; summary: string; lines: TL[] }>({
  calculatedStartTime: '',
  summary: '',
  lines: []
})

function startEdit() {
  const a = currentAgenda.value
  if (!a) return
  editableAgenda.value = {
    calculatedStartTime: a.calculatedStartTime ?? '',
    summary: a.summary ?? a.explanation ?? '',
    lines: JSON.parse(JSON.stringify(a.lines || []))
  }
  isEditing.value = true
}

function resetEdit() {
  startEdit()
}

function saveEdit() {
  const d = detail.value
  if (!d || activeIndex.value < 0) {
    isEditing.value = false
    return
  }
  const realAgenda = d.agenda?.filter(a => Array.isArray(a?.lines) && a.lines!.length > 0)[activeIndex.value]
  if (realAgenda) {
    realAgenda.calculatedStartTime = editableAgenda.value.calculatedStartTime
    realAgenda.summary = editableAgenda.value.summary
    realAgenda.lines = JSON.parse(JSON.stringify(editableAgenda.value.lines))
  }
  isEditing.value = false
}

const linesForView = computed<TL[]>(() => {
  const kw = keyword.value.trim().toLowerCase()
  const lines = isEditing.value ? editableAgenda.value.lines : currentLines.value
  if (!kw) return lines
  return lines.filter(
    l =>
      (l.speaker || '').toLowerCase().includes(kw) ||
      (l.text || '').toLowerCase().includes(kw)
  )
})


function formatAgendaTitle(a: Agenda) {
  const num = (a?.number ?? '').toString().trim()
  const title = (a?.title ?? '').toString().trim()
  return num ? `${num}  ${title}` : title
}
function toMMSS(sec?: number) {
  if (typeof sec !== 'number' || Number.isNaN(sec)) return ''
  const m = Math.floor(sec / 60)
  const s = Math.floor(sec % 60)
  return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`
}

watch(activeIndex, () => {
  isEditing.value = false
})
</script>

<style scoped>
/* ========== Layout ========== */
.layout { display: flex; height: 100vh; }
.sidebar { width: 240px; background-color: #f4f4f4; border-right: 1px solid #ccc; overflow-y: auto; }
.content { flex: 1; padding: 20px; overflow-y: auto; }

/* ========== Topbar / Tools ========== */
.topbar { display: flex; align-items: center; justify-content: flex-end; gap: 12px; margin-bottom: 10px; }
.tools { display: flex; align-items: center; gap: 12px; }
.search { width: 260px; padding: 6px 10px; border: 1px solid #cfd8e3; border-radius: 6px; outline: none; }
.search:focus { border-color: #7aa7e0; box-shadow: 0 0 0 3px rgba(122, 167, 224, 0.2); }

/* ========== Heading ========== */
.title { margin: 8px 0 16px; }
.subtitle { color: #6b7280; font-weight: 400; }

/* ========== Agenda Cards ========== */
.agenda-list, .agenda-single { display: grid; gap: 14px; }
.agenda-card { background: #fbfdff; border: 1px solid #e6eef8; border-radius: 12px; padding: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.04); }
.agenda-header { display: flex; align-items: center; gap: 10px; margin-bottom: 8px; }
.badge { width: 36px; height: 36px; border-radius: 50%; background: #e8f0fe; color: #274690; display: grid; place-items: center; font-weight: 700; }
.meta .name { font-weight: 700; }
.meta .count { font-size: 12px; color: #6b7280; }


.edit-actions { margin-left: auto; display: flex; gap: 8px; }
.btn { padding: 6px 12px; border: 1px solid #cfd8e3; background: #aabcce; border-radius: 6px; cursor: pointer; }
.btn.primary { background: #3b82f6; color: #fff; border-color: #3b82f6; }
.btn.ghost { background: #fff; }

/* ========== Editable fields ========== */
.edit-fields { display: grid; gap: 10px; }
.field > label { font-weight: 700; display: block; margin: 6px 0; }
.input { width: 100%; padding: 8px 10px; border: 1px solid #d1d5db; border-radius: 6px; }
.textarea { min-height: 80px; resize: vertical; }


.summary-divider { border-top: 1px dashed #e5e7eb; margin: 12px 0 16px; }

/* ========== Transcript Lines ========== */
.line-list { list-style: none; margin: 0; padding: 0; display: grid; gap: 8px; }
.line-item { display: grid; grid-template-columns: auto auto 1fr; gap: 8px; align-items: start; }
.time { font-size: 12px; color: #64748b; padding: 2px 6px; background: #eef2ff; border-radius: 6px; height: fit-content; }

.speaker { font-weight: 700; color: #0f172a; }
.text { white-space: pre-wrap; line-height: 1.6; }


.line-input { width: 100%; border: 1px solid #d1d5db; border-radius: 6px; padding: 6px 8px; }
.speaker-input { max-width: 160px; }
.text-input { min-height: 44px; resize: vertical; }


.empty { color: #6b7280; }
</style>
