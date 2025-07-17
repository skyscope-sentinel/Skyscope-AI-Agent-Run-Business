import {
  ClearOutlined,
  CloseOutlined,
  DeleteOutlined,
  EditOutlined,
  FolderOutlined,
  UploadOutlined
} from '@ant-design/icons'
import DragableList from '@renderer/components/DragableList'
import PromptPopup from '@renderer/components/Popups/PromptPopup'
import { useAssistant, useAssistants } from '@renderer/hooks/useAssistant'
import { useSettings } from '@renderer/hooks/useSettings'
import { TopicManager } from '@renderer/hooks/useTopic'
import { fetchMessagesSummary } from '@renderer/services/api'
import { EVENT_NAMES, EventEmitter } from '@renderer/services/event'
import store, { useAppSelector } from '@renderer/store'
import { setGenerating } from '@renderer/store/runtime'
import { Assistant, Topic } from '@renderer/types'
import { exportTopicAsMarkdown } from '@renderer/utils/export'
import { Dropdown, MenuProps } from 'antd'
import dayjs from 'dayjs'
import { findIndex } from 'lodash'
import { FC, useCallback } from 'react'
import { useTranslation } from 'react-i18next'
import styled from 'styled-components'

interface Props {
  assistant: Assistant
  activeTopic: Topic
  setActiveTopic: (topic: Topic) => void
}

const Topics: FC<Props> = ({ assistant: _assistant, activeTopic, setActiveTopic }) => {
  const { assistants } = useAssistants()
  const { assistant, removeTopic, moveTopic, updateTopic, updateTopics } = useAssistant(_assistant.id)
  const { t } = useTranslation()
  const generating = useAppSelector((state) => state.runtime.generating)
  const { showTopicTime } = useSettings()

  const borderRadius = showTopicTime ? 12 : 17

  const onDeleteTopic = useCallback(
    (topic: Topic) => {
      if (generating) {
        window.message.warning({ content: t('message.switch.disabled'), key: 'generating' })
        return
      }
      const index = findIndex(assistant.topics, (t) => t.id === topic.id)
      setActiveTopic(assistant.topics[index + 1 === assistant.topics.length ? 0 : index + 1])
      removeTopic(topic)
    },
    [assistant.topics, generating, removeTopic, setActiveTopic, t]
  )

  const onMoveTopic = useCallback(
    (topic: Topic, toAssistant: Assistant) => {
      if (generating) {
        window.message.warning({ content: t('message.switch.disabled'), key: 'generating' })
        return
      }
      const index = findIndex(assistant.topics, (t) => t.id === topic.id)
      setActiveTopic(assistant.topics[index + 1 === assistant.topics.length ? 0 : index + 1])
      moveTopic(topic, toAssistant)
    },
    [assistant.topics, generating, moveTopic, setActiveTopic, t]
  )

  const onSwitchTopic = useCallback(
    (topic: Topic) => {
      if (generating) {
        window.message.warning({ content: t('message.switch.disabled'), key: 'generating' })
        return
      }
      setActiveTopic(topic)
    },
    [generating, setActiveTopic, t]
  )

  const onClearMessages = useCallback(() => {
    window.keyv.set(EVENT_NAMES.CHAT_COMPLETION_PAUSED, true)
    store.dispatch(setGenerating(false))
    EventEmitter.emit(EVENT_NAMES.CLEAR_MESSAGES)
  }, [])

  const getTopicMenuItems = useCallback(
    (topic: Topic) => {
      const menus: MenuProps['items'] = [
        {
          label: t('chat.topics.auto_rename'),
          key: 'auto-rename',
          icon: <i className="iconfont icon-business-smart-assistant" style={{ fontSize: '14px' }} />,
          async onClick() {
            const messages = await TopicManager.getTopicMessages(topic.id)
            if (messages.length >= 2) {
              const summaryText = await fetchMessagesSummary({ messages, assistant })
              if (summaryText) {
                updateTopic({ ...topic, name: summaryText })
              }
            }
          }
        },
        {
          label: t('chat.topics.edit.title'),
          key: 'rename',
          icon: <EditOutlined />,
          async onClick() {
            const name = await PromptPopup.show({
              title: t('chat.topics.edit.title'),
              message: '',
              defaultValue: topic?.name || ''
            })
            if (name && topic?.name !== name) {
              updateTopic({ ...topic, name })
            }
          }
        },
        {
          label: t('chat.topics.clear.title'),
          key: 'clear-messages',
          icon: <ClearOutlined />,
          async onClick() {
            window.modal.confirm({
              title: t('chat.input.clear.content'),
              centered: true,
              onOk: onClearMessages
            })
          }
        },
        {
          label: t('chat.topics.export.title'),
          key: 'export',
          icon: <UploadOutlined />,
          children: [
            {
              label: t('chat.topics.export.image'),
              key: 'image',
              onClick: () => EventEmitter.emit(EVENT_NAMES.EXPORT_TOPIC_IMAGE, topic)
            },
            {
              label: t('chat.topics.export.md'),
              key: 'markdown',
              onClick: () => exportTopicAsMarkdown(topic)
            }
          ]
        }
      ]

      if (assistants.length > 1 && assistant.topics.length > 1) {
        menus.push({
          label: t('chat.topics.move_to'),
          key: 'move',
          icon: <FolderOutlined />,
          children: assistants
            .filter((a) => a.id !== assistant.id)
            .map((a) => ({
              label: a.name,
              key: a.id,
              onClick: () => onMoveTopic(topic, a)
            }))
        })
      }

      if (assistant.topics.length > 1) {
        menus.push({ type: 'divider' })
        menus.push({
          label: t('common.delete'),
          danger: true,
          key: 'delete',
          icon: <DeleteOutlined />,
          onClick: () => onDeleteTopic(topic)
        })
      }

      return menus
    },
    [assistant, assistants, onClearMessages, onDeleteTopic, onMoveTopic, t, updateTopic]
  )

  return (
    <Container>
      <DragableList list={assistant.topics} onUpdate={updateTopics}>
        {(topic) => {
          const isActive = topic.id === activeTopic?.id
          return (
            <Dropdown menu={{ items: getTopicMenuItems(topic) }} trigger={['contextMenu']} key={topic.id}>
              <TopicListItem
                className={isActive ? 'active' : ''}
                style={{ borderRadius }}
                onClick={() => onSwitchTopic(topic)}>
                <TopicName className="name">{topic.name.replace('`', '')}</TopicName>
                {showTopicTime && <TopicTime>{dayjs(topic.createdAt).format('MM/DD HH:mm')}</TopicTime>}
                {isActive && (
                  <MenuButton
                    className="menu"
                    onClick={(e) => {
                      e.stopPropagation()
                      if (assistant.topics.length === 1) {
                        return onClearMessages()
                      }
                      onDeleteTopic(topic)
                    }}>
                    <CloseOutlined />
                  </MenuButton>
                )}
              </TopicListItem>
            </Dropdown>
          )
        }}
      </DragableList>
      <div style={{ minHeight: '10px' }}></div>
    </Container>
  )
}

const Container = styled.div`
  display: flex;
  flex-direction: column;
  padding-top: 10px;
  max-height: calc(100vh - var(--navbar-height) - 70px);
`

const TopicListItem = styled.div`
  padding: 7px 12px;
  margin: 0 10px;
  border-radius: 17px;
  font-family: Ubuntu;
  font-size: 13px;
  display: flex;
  flex-direction: column;
  justify-content: space-between;
  position: relative;
  font-family: Ubuntu;
  cursor: pointer;
  .menu {
    opacity: 0;
    color: var(--color-text-3);
  }
  &:hover {
    background-color: var(--color-background-soft);
    .name {
      opacity: 1;
    }
  }
  &.active {
    background-color: var(--color-background-mute);
    .name {
      opacity: 1;
    }
    .menu {
      opacity: 1;
      background-color: var(--color-background-mute);
      &:hover {
        color: var(--color-text-2);
      }
    }
  }
`

const TopicName = styled.div`
  display: -webkit-box;
  -webkit-line-clamp: 1;
  -webkit-box-orient: vertical;
  overflow: hidden;
  font-size: 13px;
`

const TopicTime = styled.div`
  color: var(--color-text-3);
  font-size: 11px;
`

const MenuButton = styled.div`
  display: flex;
  flex-direction: row;
  justify-content: center;
  align-items: center;
  min-width: 22px;
  min-height: 22px;
  position: absolute;
  right: 8px;
  top: 6px;
  .anticon {
    font-size: 12px;
  }
`

export default Topics
