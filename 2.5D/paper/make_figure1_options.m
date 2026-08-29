#import <AppKit/AppKit.h>

static NSImage *LoadImage(NSString *terrainRoot, NSString *family, NSString *difficulty) {
    NSString *filename = [NSString stringWithFormat:@"terrain_%@_001_%@.png", family, difficulty];
    NSString *path = [[[terrainRoot stringByAppendingPathComponent:family]
        stringByAppendingPathComponent:filename] stringByStandardizingPath];
    NSImage *image = [[NSImage alloc] initWithContentsOfFile:path];
    if (image == nil) {
        @throw [NSException exceptionWithName:@"Figure1Options"
            reason:[NSString stringWithFormat:@"Không đọc được ảnh: %@", path]
            userInfo:nil];
    }
    return image;
}

static NSImage *CropPlot(NSImage *image) {
    CGImageRef source = [image CGImageForProposedRect:NULL context:nil hints:nil];
    if (source == NULL) {
        @throw [NSException exceptionWithName:@"Figure1Options"
            reason:@"Không thể đọc raster để cắt vùng đồ thị"
            userInfo:nil];
    }
    CGRect crop = CGRectMake(240, 110, 1180, 850);
    CGImageRef cropped = CGImageCreateWithImageInRect(source, crop);
    if (cropped == NULL) {
        @throw [NSException exceptionWithName:@"Figure1Options"
            reason:@"Không thể cắt vùng đồ thị"
            userInfo:nil];
    }
    NSImage *result = [[NSImage alloc] initWithCGImage:cropped size:NSMakeSize(1180, 850)];
    CGImageRelease(cropped);
    return result;
}

static void WriteGrid(
    NSString *terrainRoot,
    NSString *outputRoot,
    NSString *filename,
    NSArray<NSArray<NSArray<NSString *> *> *> *rows,
    NSInteger cellWidth
) {
    NSInteger gutter = 18;
    NSInteger margin = 24;
    NSInteger columnCount = 0;
    for (NSArray *row in rows) {
        columnCount = MAX(columnCount, (NSInteger)row.count);
    }
    NSInteger cellHeight = lround((double)cellWidth * 1080.0 / 1560.0);
    NSInteger canvasWidth = margin * 2 + columnCount * cellWidth
        + MAX(0, columnCount - 1) * gutter;
    NSInteger canvasHeight = margin * 2 + rows.count * cellHeight
        + MAX(0, (NSInteger)rows.count - 1) * gutter;

    NSBitmapImageRep *bitmap = [[NSBitmapImageRep alloc]
        initWithBitmapDataPlanes:NULL
        pixelsWide:canvasWidth
        pixelsHigh:canvasHeight
        bitsPerSample:8
        samplesPerPixel:4
        hasAlpha:YES
        isPlanar:NO
        colorSpaceName:NSDeviceRGBColorSpace
        bytesPerRow:0
        bitsPerPixel:0];
    NSGraphicsContext *context = [NSGraphicsContext graphicsContextWithBitmapImageRep:bitmap];
    [NSGraphicsContext saveGraphicsState];
    [NSGraphicsContext setCurrentContext:context];
    context.imageInterpolation = NSImageInterpolationHigh;
    [[NSColor colorWithCalibratedRed:0.933 green:0.949 blue:0.949 alpha:1.0] setFill];
    NSRectFill(NSMakeRect(0, 0, canvasWidth, canvasHeight));

    [rows enumerateObjectsUsingBlock:^(NSArray<NSArray<NSString *> *> *row, NSUInteger rowIndex, BOOL *stop) {
        NSInteger rowWidth = row.count * cellWidth + MAX(0, (NSInteger)row.count - 1) * gutter;
        NSInteger startX = (canvasWidth - rowWidth) / 2;
        NSInteger y = canvasHeight - margin - ((NSInteger)rowIndex + 1) * cellHeight
            - (NSInteger)rowIndex * gutter;
        [row enumerateObjectsUsingBlock:^(NSArray<NSString *> *item, NSUInteger columnIndex, BOOL *innerStop) {
            NSImage *image = LoadImage(terrainRoot, item[0], item[1]);
            NSInteger x = startX + (NSInteger)columnIndex * (cellWidth + gutter);
            [image drawInRect:NSMakeRect(x, y, cellWidth, cellHeight)
                fromRect:NSZeroRect
                operation:NSCompositingOperationCopy
                fraction:1.0
                respectFlipped:YES
                hints:@{NSImageHintInterpolation: @(NSImageInterpolationHigh)}];
        }];
    }];
    [context flushGraphics];
    [NSGraphicsContext restoreGraphicsState];

    NSData *png = [bitmap representationUsingType:NSBitmapImageFileTypePNG properties:@{}];
    NSString *outputPath = [outputRoot stringByAppendingPathComponent:filename];
    if (![png writeToFile:outputPath atomically:YES]) {
        @throw [NSException exceptionWithName:@"Figure1Options"
            reason:[NSString stringWithFormat:@"Không ghi được ảnh: %@", outputPath]
            userInfo:nil];
    }
}

static void DrawCenteredText(NSString *text, NSRect rect, CGFloat size, BOOL bold) {
    NSMutableParagraphStyle *style = [[NSMutableParagraphStyle alloc] init];
    style.alignment = NSTextAlignmentCenter;
    NSDictionary *attributes = @{
        NSFontAttributeName: bold
            ? [NSFont boldSystemFontOfSize:size]
            : [NSFont systemFontOfSize:size],
        NSForegroundColorAttributeName: [NSColor colorWithCalibratedWhite:0.14 alpha:1.0],
        NSParagraphStyleAttributeName: style,
    };
    NSAttributedString *label = [[NSAttributedString alloc] initWithString:text attributes:attributes];
    NSSize labelSize = label.size;
    NSRect target = NSMakeRect(
        rect.origin.x,
        rect.origin.y + (rect.size.height - labelSize.height) / 2.0,
        rect.size.width,
        labelSize.height
    );
    [label drawInRect:target];
}

static void WriteCleanGrid(
    NSString *terrainRoot,
    NSString *outputRoot,
    NSArray<NSString *> *families,
    NSString *filename,
    NSArray<NSString *> *difficultyLabels,
    NSString *legend
) {
    NSArray<NSString *> *difficulties = @[@"easy", @"medium", @"hard"];
    NSDictionary<NSString *, NSString *> *familyLabels = @{
        @"smooth_obstacles": @"Smooth\nobstacles",
        @"rolling": @"Rolling",
        @"mountain": @"Mountain",
        @"rugged": @"Rugged",
        @"plateau": @"Plateau",
    };
    NSInteger margin = 24;
    NSInteger rowLabelWidth = 180;
    NSInteger labelGutter = 12;
    NSInteger cellWidth = 700;
    NSInteger cellHeight = 504;
    NSInteger gutter = 12;
    NSInteger headerHeight = 66;
    NSInteger legendHeight = 54;
    NSInteger canvasWidth = margin * 2 + rowLabelWidth + labelGutter
        + difficulties.count * cellWidth + (difficulties.count - 1) * gutter;
    NSInteger canvasHeight = margin * 2 + headerHeight + legendHeight
        + families.count * cellHeight + (families.count - 1) * gutter;

    NSBitmapImageRep *bitmap = [[NSBitmapImageRep alloc]
        initWithBitmapDataPlanes:NULL
        pixelsWide:canvasWidth
        pixelsHigh:canvasHeight
        bitsPerSample:8
        samplesPerPixel:4
        hasAlpha:YES
        isPlanar:NO
        colorSpaceName:NSDeviceRGBColorSpace
        bytesPerRow:0
        bitsPerPixel:0];
    NSGraphicsContext *context = [NSGraphicsContext graphicsContextWithBitmapImageRep:bitmap];
    [NSGraphicsContext saveGraphicsState];
    [NSGraphicsContext setCurrentContext:context];
    context.imageInterpolation = NSImageInterpolationHigh;
    [[NSColor colorWithCalibratedRed:0.933 green:0.949 blue:0.949 alpha:1.0] setFill];
    NSRectFill(NSMakeRect(0, 0, canvasWidth, canvasHeight));

    NSInteger gridX = margin + rowLabelWidth + labelGutter;
    for (NSInteger row = 0; row < (NSInteger)families.count; row++) {
        NSString *family = families[row];
        NSInteger y = canvasHeight - margin - headerHeight
            - (row + 1) * cellHeight - row * gutter;
        for (NSInteger column = 0; column < (NSInteger)difficulties.count; column++) {
            NSString *difficulty = difficulties[column];
            NSImage *image = CropPlot(LoadImage(terrainRoot, family, difficulty));
            NSInteger x = gridX + column * (cellWidth + gutter);
            [image drawInRect:NSMakeRect(x, y, cellWidth, cellHeight)
                fromRect:NSZeroRect
                operation:NSCompositingOperationCopy
                fraction:1.0
                respectFlipped:YES
                hints:@{NSImageHintInterpolation: @(NSImageInterpolationHigh)}];
        }
        DrawCenteredText(
            familyLabels[family],
            NSMakeRect(margin, y, rowLabelWidth, cellHeight),
            26,
            YES
        );
    }
    NSInteger headerY = canvasHeight - margin - headerHeight;
    for (NSInteger column = 0; column < (NSInteger)difficulties.count; column++) {
        NSInteger x = gridX + column * (cellWidth + gutter);
        DrawCenteredText(difficultyLabels[column], NSMakeRect(x, headerY, cellWidth, headerHeight), 32, YES);
    }
    DrawCenteredText(
        legend,
        NSMakeRect(gridX, margin, 3 * cellWidth + 2 * gutter, legendHeight),
        23,
        NO
    );
    [context flushGraphics];
    [NSGraphicsContext restoreGraphicsState];

    NSData *png = [bitmap representationUsingType:NSBitmapImageFileTypePNG properties:@{}];
    NSString *outputPath = [outputRoot stringByAppendingPathComponent:filename];
    if (![png writeToFile:outputPath atomically:YES]) {
        @throw [NSException exceptionWithName:@"Figure1Options"
            reason:[NSString stringWithFormat:@"Không ghi được ảnh: %@", outputPath]
            userInfo:nil];
    }
}

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        if (argc != 2) {
            fprintf(stderr, "Usage: make_figure1_options PAPER_ROOT\n");
            return 2;
        }
        NSString *paperRoot = [[[NSString alloc] initWithUTF8String:argv[1]] stringByStandardizingPath];
        NSString *projectRoot = [paperRoot stringByDeletingLastPathComponent];
        NSString *terrainRoot = [projectRoot stringByAppendingPathComponent:@"dataset_5010_v1/images/terrain"];
        NSString *outputRoot = [paperRoot stringByAppendingPathComponent:@"figure1_options"];
        [[NSFileManager defaultManager] createDirectoryAtPath:outputRoot
            withIntermediateDirectories:YES attributes:nil error:NULL];

        NSArray<NSString *> *families = @[
            @"smooth_obstacles", @"rolling", @"mountain", @"rugged", @"plateau"
        ];
        NSArray<NSString *> *difficulties = @[@"easy", @"medium", @"hard"];
        NSMutableArray *allRows = [NSMutableArray array];
        for (NSString *family in families) {
            NSMutableArray *row = [NSMutableArray array];
            for (NSString *difficulty in difficulties) {
                [row addObject:@[family, difficulty]];
            }
            [allRows addObject:row];
        }

        WriteGrid(terrainRoot, outputRoot, @"option_a_compact_15_panels.png", allRows, 780);
        WriteGrid(terrainRoot, outputRoot, @"option_b1_split_first_9_panels.png",
            [allRows subarrayWithRange:NSMakeRange(0, 3)], 780);
        WriteGrid(terrainRoot, outputRoot, @"option_b2_split_last_6_panels.png",
            [allRows subarrayWithRange:NSMakeRange(3, 2)], 780);

        NSMutableArray *endpointRows = [NSMutableArray array];
        for (NSString *family in families) {
            [endpointRows addObject:@[@[family, @"easy"], @[family, @"hard"]]];
        }
        WriteGrid(terrainRoot, outputRoot, @"option_c_easy_hard_10_panels.png",
            endpointRows, 1040);
        WriteCleanGrid(
            terrainRoot,
            outputRoot,
            families,
            @"option_d_clean_15_panels.png",
            @[@"Dễ", @"Trung bình", @"Khó"],
            @"S: Start    G: Goal    Vùng xám: tâm robot không thể đi qua"
        );
        WriteCleanGrid(
            terrainRoot,
            outputRoot,
            families,
            @"option_d_clean_15_panels_en.png",
            @[@"Easy", @"Medium", @"Hard"],
            @"S: Start    G: Goal    Grey: reference robot centre is blocked"
        );
        printf("Đã tạo các phương án Hình 1 tại %s\n", outputRoot.UTF8String);
    }
    return 0;
}
