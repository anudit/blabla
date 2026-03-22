import { signal } from '@preact/signals';

export const isPlayingSignal            = signal(false);
export const playbackStateSignal        = signal<string>("Idle");
export const ttsStatusSignal            = signal<string>("Init");
export const isModelReadySignal         = signal(false);
export const currentSentenceIndexSignal = signal(-1);
export const restartSignal              = signal(0);

// Document State
export const sentencesSignal            = signal<any[]>([]);
export const fileTypeSignal             = signal<'pdf' | 'epub' | 'text' | null>(null);
export const playbackSpeedSignal        = signal(1.0);
export const selectedVoiceSignal        = signal('Bella');
export const currentFileIdSignal        = signal<string | null>(null);
export const currentFileNameSignal      = signal<string | null>(null);
export const outlineSignal              = signal<any[]>([]);

if (process.env.NODE_ENV !== 'production') {
  // @ts-ignore
  import('signals-devtool-provider').then(({ default: signalsDevtool }) => {
    import('@preact/signals').then(({ effect }) => {
      signalsDevtool.init({
        signals: {
          isPlayingSignal,
          playbackStateSignal,
          ttsStatusSignal,
          isModelReadySignal,
          currentSentenceIndexSignal,
          restartSignal,
          sentencesSignal,
          fileTypeSignal,
          playbackSpeedSignal,
          selectedVoiceSignal,
          currentFileIdSignal,
          currentFileNameSignal,
          outlineSignal,
        },
        effect,
      });
    });
  });
}
