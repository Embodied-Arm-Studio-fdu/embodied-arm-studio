#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <libavcodec/avcodec.h>
#include <libavutil/error.h>
#include <libavutil/imgutils.h>
#include <libavutil/opt.h>
#include <libswscale/swscale.h>

static int read_exact(void *buffer, size_t size) {
    size_t offset = 0;
    while (offset < size) {
        size_t got = fread((uint8_t *)buffer + offset, 1, size - offset, stdin);
        if (got == 0) return feof(stdin) ? 0 : -1;
        offset += got;
    }
    return 1;
}

static int write_packet(const AVPacket *packet) {
    const char magic[4] = {'P','K','T','1'};
    uint32_t size = (uint32_t)packet->size;
    int64_t pts = packet->pts, dts = packet->dts;
    int32_t flags = packet->flags;
    if (fwrite(magic, 1, 4, stdout) != 4 ||
        fwrite(&size, sizeof(size), 1, stdout) != 1 ||
        fwrite(&pts, sizeof(pts), 1, stdout) != 1 ||
        fwrite(&dts, sizeof(dts), 1, stdout) != 1 ||
        fwrite(&flags, sizeof(flags), 1, stdout) != 1 ||
        fwrite(packet->data, 1, packet->size, stdout) != (size_t)packet->size)
        return -1;
    return 0;
}

static int encode_and_respond(AVCodecContext *context, AVFrame *frame) {
    AVPacket packet;
    AVPacket *packets = NULL;
    uint32_t count = 0, capacity = 0, index;
    int result = avcodec_send_frame(context, frame);
    if (result < 0) return result;
    av_init_packet(&packet);
    while ((result = avcodec_receive_packet(context, &packet)) >= 0) {
        if (count == capacity) {
            capacity = capacity ? capacity * 2 : 2;
            AVPacket *grown = realloc(packets, capacity * sizeof(*packets));
            if (!grown) return AVERROR(ENOMEM);
            packets = grown;
        }
        packets[count] = packet;
        memset(&packet, 0, sizeof(packet));
        count++;
    }
    if (result != AVERROR(EAGAIN) && result != AVERROR_EOF) return result;
    if (fwrite("RSP1", 1, 4, stdout) != 4 || fwrite(&count, 4, 1, stdout) != 1)
        return AVERROR(EIO);
    for (index = 0; index < count; index++) {
        if (write_packet(&packets[index]) < 0) return AVERROR(EIO);
        av_packet_unref(&packets[index]);
    }
    free(packets);
    fflush(stdout);
    return 0;
}

int main(int argc, char **argv) {
    const AVCodec *codec;
    AVCodecContext *context = NULL;
    AVFrame *frame = NULL;
    struct SwsContext *sws = NULL;
    uint8_t *input = NULL;
    int width, height, fps, bitrate, gop, gpu, result = 1;
    char error[AV_ERROR_MAX_STRING_SIZE];
    if (argc != 7) {
        fprintf(stderr, "usage: nvenc_helper width height fps bitrate gop gpu\n");
        return 2;
    }
    width = atoi(argv[1]); height = atoi(argv[2]); fps = atoi(argv[3]);
    bitrate = atoi(argv[4]); gop = atoi(argv[5]); gpu = atoi(argv[6]);
    codec = avcodec_find_encoder_by_name("h264_nvenc");
    if (!codec) { fprintf(stderr, "h264_nvenc encoder not compiled\n"); return 3; }
    context = avcodec_alloc_context3(codec);
    frame = av_frame_alloc();
    if (!context || !frame) goto cleanup;
    context->width = width; context->height = height;
    context->time_base = (AVRational){1, 1000000};
    context->framerate = (AVRational){fps, 1};
    context->pix_fmt = AV_PIX_FMT_NV12;
    context->bit_rate = bitrate; context->gop_size = gop; context->max_b_frames = 0;
    av_opt_set(context->priv_data, "zerolatency", "1", 0);
    av_opt_set(context->priv_data, "delay", "0", 0);
    av_opt_set_int(context->priv_data, "gpu", gpu, 0);
    if ((result = avcodec_open2(context, codec, NULL)) < 0) goto fail;
    fprintf(stderr, "NVENC_INITIALIZED encoder=%s input=nv12\n", codec->name);
    frame->format = context->pix_fmt; frame->width = width; frame->height = height;
    if ((result = av_frame_get_buffer(frame, 32)) < 0) goto fail;
    sws = sws_getContext(width, height, AV_PIX_FMT_BGRA, width, height,
                         AV_PIX_FMT_NV12, SWS_FAST_BILINEAR, NULL, NULL, NULL);
    input = malloc((size_t)width * height * 4);
    if (!sws || !input) goto cleanup;
    for (;;) {
        char magic[4]; int64_t pts_us; int read = read_exact(magic, 4);
        if (read == 0) break;
        if (read < 0 || memcmp(magic, "FRM1", 4) != 0 ||
            read_exact(&pts_us, sizeof(pts_us)) != 1 ||
            read_exact(input, (size_t)width * height * 4) != 1) {
            fprintf(stderr, "invalid/truncated frame protocol\n"); goto cleanup;
        }
        if ((result = av_frame_make_writable(frame)) < 0) goto fail;
        const uint8_t *source[4] = {input, NULL, NULL, NULL};
        int source_stride[4] = {width * 4, 0, 0, 0};
        sws_scale(sws, source, source_stride, 0, height, frame->data, frame->linesize);
        frame->pts = pts_us;
        if ((result = encode_and_respond(context, frame)) < 0) goto fail;
    }
    if ((result = encode_and_respond(context, NULL)) < 0) goto fail;
    result = 0;
    goto cleanup;
fail:
    av_strerror(result, error, sizeof(error));
    fprintf(stderr, "NVENC_ERROR %s\n", error);
cleanup:
    free(input); sws_freeContext(sws); av_frame_free(&frame);
    avcodec_free_context(&context);
    return result == 0 ? 0 : 1;
}
