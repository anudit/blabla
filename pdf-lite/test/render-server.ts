const fixturePath = process.env.PDF_LITE_FIXTURE || './test/pdf-sample_0.pdf';
const expectedText = process.env.PDF_LITE_EXPECTED_TEXT || (
  fixturePath.endsWith('pdf-sample_0.pdf')
    ? 'Dummy PDF file'
    : fixturePath.endsWith('test2.pdf')
      ? 'Reinforcement learning-based dynamic load balancing in edge computing networks'
      : ''
);
const fixture = new Uint8Array(await Bun.file(fixturePath).arrayBuffer());
const fixtureBase64 = Buffer.from(fixture).toString('base64');
const pageNumber = process.env.PDF_LITE_PAGE || '1';
const html = (await Bun.file('./test/render.html').text())
  .replace('__PDF_BASE64__', fixtureBase64)
  .replace('__PAGE_NUMBER__', pageNumber)
  .replace('__EXPECTED_TEXT__', expectedText.replace(/'/g, "\\'"));

const server = Bun.serve({
  port: Number(process.env.PDF_LITE_PORT || 0),
  routes: {
    '/': () => new Response(html, { headers: { 'Content-Type': 'text/html' } }),
    '/dist/pdf-lite.js': () => new Response(Bun.file('./dist/pdf-lite.js'), { headers: { 'Content-Type': 'application/javascript' } }),
  },
});
console.log(`pdf-lite render test: http://127.0.0.1:${server.port}/`);
