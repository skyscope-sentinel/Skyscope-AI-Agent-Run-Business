import { isMac } from '@renderer/config/constant'
import { isLocalAi } from '@renderer/config/env'
import db from '@renderer/databases'
import i18n from '@renderer/i18n'
import { useAppDispatch } from '@renderer/store'
import { setAvatar } from '@renderer/store/runtime'
import { runAsyncFunction } from '@renderer/utils'
import { useLiveQuery } from 'dexie-react-hooks'
import { useEffect } from 'react'

import { useDefaultModel } from './useAssistant'
import { useSettings } from './useSettings'
import { useRuntime } from './useStore'

export function useAppInit() {
  const dispatch = useAppDispatch()
  const { proxyUrl, language, windowStyle, manualUpdateCheck } = useSettings()
  const { minappShow } = useRuntime()
  const { setDefaultModel, setTopicNamingModel, setTranslateModel } = useDefaultModel()
  const avatar = useLiveQuery(() => db.settings.get('image://avatar'))

  useEffect(() => {
    avatar?.value && dispatch(setAvatar(avatar.value))
  }, [avatar, dispatch])

  useEffect(() => {
    document.getElementById('spinner')?.remove()
    runAsyncFunction(async () => {
      const { isPackaged } = await window.api.getAppInfo()
      if (isPackaged && !manualUpdateCheck) {
        setTimeout(window.api.checkForUpdate, 3000)
      }
    })
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  useEffect(() => {
    proxyUrl && window.api.setProxy(proxyUrl)
  }, [proxyUrl])

  useEffect(() => {
    i18n.changeLanguage(language || navigator.language || 'en-US')
  }, [language])

  useEffect(() => {
    const transparentWindow = windowStyle === 'transparent' && isMac && !minappShow
    window.root.style.background = transparentWindow ? 'var(--navbar-background-mac)' : 'var(--navbar-background)'
    !isMac && import('@renderer/assets/styles/scrollbar.scss')
  }, [windowStyle, minappShow])

  useEffect(() => {
    if (isLocalAi) {
      const model = JSON.parse(import.meta.env.VITE_RENDERER_INTEGRATED_MODEL)
      setDefaultModel(model)
      setTopicNamingModel(model)
      setTranslateModel(model)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])
}
