import adapter from '@sveltejs/adapter-static';

/** @type {import('@sveltejs/kit').Config} */
const config = {
  kit: {
    adapter: adapter({
      strict: false
    }),
    paths: {
      relative: true
    },
    prerender: {
      handleHttpError: 'warn'
    }
  }
};

export default config;
