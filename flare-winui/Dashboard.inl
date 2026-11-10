// SPDX-License-Identifier: Apache-2.0
    std::wstring stage_label(std::wstring const& stage) {
        if(stage==L"verify_stored_data")return L"Check stored core data";
        if(stage==L"prepare_model")return L"Inspect prepared core";
        if(stage==L"initialize_gpu")return L"Initialize GPU";
        if(stage==L"load_weights")return L"Load model data";
        if(stage==L"prepare_kernels"||stage==L"prepare_execution")return L"Load core";
        if(stage==L"plan_cache"||stage==L"allocate_cache"||stage==L"prepare_host_prefix_cache")return L"Allocate KV and prefix caches";
        if(stage==L"allocate_model_memory")return L"Allocate model memory";
        if(stage==L"prepare_embedding_tables")return L"Load embedding tables";
        if(stage==L"load_tokenizer")return L"Load tokenizer and frontend";
        if(stage==L"start_server")return L"Check the local server";
        if(stage==L"shutdown_server")return L"Stop local serving";
        return L"Prepare local serving";
    }
    TextBlock recommendation_badge() {
        auto result=label(L"Recommended",11);
        result.Foreground(theme_resources_.Lookup(box_value(L"AppAccentBrush")).as<Microsoft::UI::Xaml::Media::Brush>());
        return result;
    }
    Border home_model_card(ModelItem model,std::wstring action_id=L"",bool include_details=false) {
        Grid row;row.ColumnSpacing(12);ColumnDefinition icon;icon.Width(GridLength{44,GridUnitType::Pixel});row.ColumnDefinitions().Append(icon);row.ColumnDefinitions().Append(ColumnDefinition());ColumnDefinition actions_column;actions_column.Width(GridLength{1,GridUnitType::Auto});row.ColumnDefinitions().Append(actions_column);
        auto mark=model_mark(model.name);mark.VerticalAlignment(VerticalAlignment::Center);row.Children().Append(mark);
        StackPanel info;info.Spacing(5);Grid::SetColumn(info,1);info.VerticalAlignment(VerticalAlignment::Center);
        if(model.recommended())info.Children().Append(recommendation_badge());
        auto name=label(model.name,15);name.MaxLines(2);name.FontWeight(Windows::UI::Text::FontWeights::SemiBold());info.Children().Append(name);
        auto metadata=model.available()?(model.size_known?bytes(model.size)+L" · ":L"")+(model.flags&FLARE_MODEL_LOCAL?L"Downloaded":L"On this PC"):L"";
        if(!model.available()&&(model.metadata_known&FLARE_MODEL_REVISION_SIZE))metadata=bytes(model.revision_bytes)+L" repository files";
        if(!model.error.empty())metadata=L"Needs attention";
        if(!metadata.empty()){auto caption=label(metadata,12,true);caption.MaxLines(2);info.Children().Append(caption);}row.Children().Append(info);
        StackPanel actions;actions.Orientation(Orientation::Horizontal);actions.Spacing(8);actions.VerticalAlignment(VerticalAlignment::Center);Grid::SetColumn(actions,2);
        if(include_details){Button details;details.Content(box_value(L"Details"));details.Style(theme_resources_.Lookup(box_value(L"TextAction")).as<Style>());details.Click([this,model](auto const&,auto const&){model_details(model);});actions.Children().Append(details);}
        Button use;use.Content(box_value(model.flags&FLARE_MODEL_ACTIVE?L"In use":model.available()?L"Run model":home_inventory_pending_?L"Checking…":L"Download"));use.Style(control<Button>(L"ChooseModelButton").Style());
        use.IsEnabled((model.available()?can_change_model_:!home_inventory_pending_)&&model.error.empty()&&!(model.flags&FLARE_MODEL_ACTIVE));
        Automation::AutomationProperties::SetName(use,(model.available()?L"Run ":L"Download ")+model.name);if(!action_id.empty())Automation::AutomationProperties::SetAutomationId(use,action_id);
        use.Click([this,model](auto const&,auto const&){if(model.available())activate(model.id,model.name);else download_repository(model.source);});actions.Children().Append(use);row.Children().Append(actions);
        StackPanel body;body.Children().Append(row);auto result=card(body);if(!model.error.empty())ToolTipService::SetToolTip(result,box_value(model.error));return result;
    }
    void render_home_models() {
        if(!root_)return;
        std::vector<ModelItem const*> local,recommended;
        auto fingerprint=std::to_wstring(can_change_model_)+L"/"+std::to_wstring(current_operation_)+L"/"+std::to_wstring(session_state_)+L"/"+std::to_wstring(home_inventory_pending_)+L"/"+std::to_wstring(card_padding_);
        for(auto const& model:models_)if((model.available()&&!model.catalog())||(model.catalog()&&model.recommended())){
            if(model.catalog())recommended.push_back(&model);else local.push_back(&model);
            for(auto const& field:{std::wstring(to_hstring(model.id)),model.name,model.source,model.error})fingerprint+=std::to_wstring(field.size())+L":"+field;
            fingerprint+=L"/"+std::to_wstring(model.flags)+L"/"+std::to_wstring(model.size)+L"/"+std::to_wstring(model.size_known);
            fingerprint+=L"/"+std::to_wstring(model.metadata_known)+L"/"+std::to_wstring(model.revision_bytes);
        }
        if(fingerprint==home_model_fingerprint_)return;
        home_model_fingerprint_=std::move(fingerprint);
        std::stable_sort(local.begin(),local.end(),[](auto a,auto b){return a->recommended()&&!b->recommended();});
        auto list=control<StackPanel>(L"HomeModelList");list.Children().Clear();
        auto recommendations=control<StackPanel>(L"HomeRecommendedList");recommendations.Children().Clear();
        control<StackPanel>(L"HomeLocalSection").Visibility(!local.empty()||home_inventory_pending_?Visibility::Visible:Visibility::Collapsed);
        control<StackPanel>(L"HomeRecommendedSection").Visibility(local.empty()&&!recommended.empty()?Visibility::Visible:Visibility::Collapsed);
        control<TextBlock>(L"HomeLibraryNote").Visibility(local.empty()&&home_inventory_pending_?Visibility::Visible:Visibility::Collapsed);
        control<Border>(L"HomeLibraryEmpty").Visibility(local.empty()&&recommended.empty()&&!home_inventory_pending_?Visibility::Visible:Visibility::Collapsed);
        control<StackPanel>(L"HomeLibraryActions").Visibility(local.empty()&&recommended.empty()?Visibility::Collapsed:Visibility::Visible);
        control<Button>(L"HomeAllModelsButton").Visibility(local.size()>3?Visibility::Visible:Visibility::Collapsed);
        for(size_t i=0;i<std::min(local.size(),size_t{3});++i)list.Children().Append(home_model_card(*local[i],local[i]->recommended()?L"HomeRecommendedAction"+std::to_wstring(i):L"HomeLocalAction"+std::to_wstring(i)));
        if(local.empty())for(size_t i=0;i<recommended.size();++i)recommendations.Children().Append(home_model_card(*recommended[i],L"HomeRecommendedAction"+std::to_wstring(i)));
    }
    void update_home(flare_status const& value) {
        bool ready=value.state==FLARE_READY&&value.accepting;
        home_inventory_pending_=value.busy&&value.state==FLARE_IDLE;
        control<Grid>(L"HomeActions").Visibility(ready?Visibility::Visible:Visibility::Collapsed);
        control<StackPanel>(L"HomeSetup").Visibility(value.state==FLARE_IDLE||value.state==FLARE_FAILED?Visibility::Visible:Visibility::Collapsed);
        control<TextBlock>(L"HomeSubtitle").Visibility(ready||value.state==FLARE_IDLE?Visibility::Collapsed:Visibility::Visible);
        control<TextBlock>(L"Readiness").Visibility(value.state==FLARE_IDLE?Visibility::Collapsed:Visibility::Visible);
        std::wstring subtitle;
        if(value.state==FLARE_LOADING)subtitle=L"Loading the prepared core.";
        else if(value.state==FLARE_DRAINING)subtitle=L"Accepted requests are finishing before the model changes.";
        else if(value.state==FLARE_STOPPING)subtitle=L"Flare is stopping local serving.";
        else if(value.state==FLARE_REVIEW_REQUIRED)subtitle=L"Review the publisher and license before Flare loads this core.";
        else if(value.state==FLARE_FAILED)subtitle=L"Review the error, retry, or choose another model.";
        else if(value.state==FLARE_RESTART_REQUIRED)subtitle=L"Restart Flare before loading a model again. Review the error details above.";
        text(L"HomeSubtitle",subtitle);
    }
    void update_loading(flare_status const& value) {
        bool working=value.state==FLARE_LOADING||value.state==FLARE_DRAINING||value.state==FLARE_STOPPING||value.state==FLARE_REVIEW_REQUIRED;
        bool failed=value.state==FLARE_FAILED&&!pending_model_id_.empty();
        control<Border>(L"LoadingCard").Visibility(working||failed?Visibility::Visible:Visibility::Collapsed);
        control<Border>(L"MetricsCard").Visibility(value.has_metrics?Visibility::Visible:Visibility::Collapsed);
        control<Grid>(L"RuntimeCachePanels").Visibility(value.has_metrics?Visibility::Visible:Visibility::Collapsed);
        text(L"LoadingNote",value.state==FLARE_REVIEW_REQUIRED?L"Review the exact publisher and license before continuing.":value.state==FLARE_DRAINING?L"Finishing accepted requests before releasing the model.":failed?wide(value.error):value.detail.length?wide(value.detail):L"Preparing local serving.");
        auto current=value.state==FLARE_DRAINING?L"Finish accepted requests":value.state==FLARE_REVIEW_REQUIRED?L"Review publisher and license":value.state==FLARE_STOPPING?L"Stop local serving":stage_label(wide(value.stage));
        if(working&&(observed_steps_.empty()||observed_steps_.back()!=current)){observed_steps_.push_back(current);if(observed_steps_.size()>8)observed_steps_.erase(observed_steps_.begin());}
        auto signature=current+std::to_wstring(value.state)+std::to_wstring(observed_steps_.size());
        if(signature!=loading_fingerprint_){loading_fingerprint_=signature;auto steps=control<StackPanel>(L"LoadingSteps");steps.Children().Clear();for(size_t i=0;i<observed_steps_.size();++i){StackPanel row;row.Orientation(Orientation::Horizontal);row.Spacing(12);row.Children().Append(label(i+1<observed_steps_.size()?L"✓":std::to_wstring(i+1),12,true));row.Children().Append(label(observed_steps_[i],14));steps.Children().Append(row);}}
        auto progress=control<ProgressBar>(L"LoadProgress");progress.Visibility(working&&value.state!=FLARE_REVIEW_REQUIRED?Visibility::Visible:Visibility::Collapsed);progress.IsIndeterminate(!value.stage_total_known||!value.stage_total);progress.Maximum(100);if(value.stage_total_known&&value.stage_total)progress.Value(std::min(100.0,100.0*value.stage_completed/value.stage_total));
        control<Button>(L"CancelSwitchButton").Visibility((value.state==FLARE_DRAINING||value.state==FLARE_LOADING)&&value.current_operation?Visibility::Visible:Visibility::Collapsed);
        control<Button>(L"CancelSwitchButton").Content(box_value(value.state==FLARE_DRAINING?L"Cancel switch":L"Cancel loading"));
        control<Button>(L"RetryModelButton").Visibility(failed?Visibility::Visible:Visibility::Collapsed);
    }
    void render_status(flare_status const& value) {
        session_state_=value.state;current_operation_=value.current_operation;active_requests_=value.active_requests;waiting_requests_=value.waiting_requests;
        active_name_=wide(value.active_model);model_directory_=wide(value.model_directory);endpoint_=wide(value.endpoint);
        can_change_model_=!value.busy&&(value.state==FLARE_IDLE||value.state==FLARE_READY||value.state==FLARE_FAILED);
        control<Button>(L"ChooseModelButton").IsEnabled(can_change_model_);
        bool transition=value.state==FLARE_LOADING||value.state==FLARE_DRAINING||value.state==FLARE_REVIEW_REQUIRED;
        text(L"ActiveModel",value.state==FLARE_IDLE?L"Run a model":transition&&!pending_model_name_.empty()?pending_model_name_:!active_name_.empty()?active_name_:!pending_model_name_.empty()?pending_model_name_:L"Run a model");
        bool ready=value.state==FLARE_READY&&value.accepting;
        text(L"Readiness",ready?(value.active_requests?L"● Running":L"● Ready"):value.state==FLARE_IDLE?L"No model loaded":state_name(value.state));
        control<TextBlock>(L"Readiness").Foreground(theme_resources_.Lookup(box_value(ready?L"PositiveBrush":L"SecondaryTextBrush")).as<Microsoft::UI::Xaml::Media::Brush>());
        control<Button>(L"ChooseModelButton").Content(box_value(value.state==FLARE_READY?L"Switch model":L"Run a model"));
        control<Button>(L"StopButton").IsEnabled(value.state==FLARE_READY||value.state==FLARE_LOADING);
        text(L"ModelDirectory",model_directory_);text(L"DownloadsDirectory",model_directory_);
        text(L"Endpoint",endpoint_.empty()?L"":endpoint_+L"/v1");
        control<Button>(L"CopyEndpointButton").IsEnabled(ready&&!endpoint_.empty());
        text(L"EndpointCopyStatus",copied_endpoint_==endpoint_&&std::chrono::steady_clock::now()<endpoint_copy_until_?L"Copied":L"");
        control<Button>(L"SearchButton").IsEnabled(!value.catalog_loading);
        text(L"CatalogState",value.catalog_loading?L"Searching Hugging Face…":catalog_requested_?L"Discovery results":L"Search to discover cores");
        update_loading(value);update_density(value);update_home(value);render_home_models();
        if(taskbar_){taskbar_->SetProgressState(hwnd_,value.state==FLARE_LOADING?(value.stage_total_known&&value.stage_total?TBPF_NORMAL:TBPF_INDETERMINATE):TBPF_NOPROGRESS);if(value.state==FLARE_LOADING&&value.stage_total_known&&value.stage_total)taskbar_->SetProgressValue(hwnd_,value.stage_completed,value.stage_total);}
    }
    void render_dashboard(flare_snapshot const* snapshot,flare_status const& status) {
        dashboard_={};
        flare_snapshot_dashboard(snapshot,&dashboard_);
        dashboard_points_.clear();
        auto count=flare_snapshot_dashboard_point_count(snapshot);
        dashboard_points_.reserve(count);
        for(size_t i=0;i<count;++i){flare_dashboard_point point{};if(flare_snapshot_dashboard_point(snapshot,i,&point))dashboard_points_.push_back(point);}
        auto const& d=dashboard_;
        if(!d.available){draw_chart();return;}
        text(L"PrefillMetric",number(d.prefill_tokens_per_second));
        text(L"GenerationMetric",number(d.generated_tokens_per_second));
        text(L"VerifyMetric",number(d.verify_steps_per_second));
        text(L"DraftMetric",d.known_fields&FLARE_DASHBOARD_ACCEPTANCE?number(d.acceptance,2):L"—");
        text(L"DepthMetric",L"┄ depth: "+(d.known_fields&FLARE_DASHBOARD_DEPTH?number(d.depth,2):L"—"));
        text(L"WeightBandwidthMetric",L"Weight bandwidth (estimate): "+(d.known_fields&FLARE_DASHBOARD_WEIGHT_BANDWIDTH?number(d.weight_bytes_per_second/1e9)+L" GB/s":L"—"));
        ToolTipService::SetToolTip(control<TextBlock>(L"WeightBandwidthMetric"),box_value(L"Verification steps/s × declared decode weight bytes. This is an estimate, not measured GPU memory throughput."));
        ToolTipService::SetToolTip(control<TextBlock>(L"DraftMetric"),box_value(L"Accepted drafts per draft-offering request round over the rate window. Depth is allocated drafts per progressing verification request round. No rounds is shown as —."));
        text(L"RequestsMetric",count_text(d.active));text(L"QueueMetric",count_text(d.pending));
        text(L"IngressMetric",count_text(d.ingress));text(L"CompletedMetric",count_text(d.completed));text(L"FailedMetric",count_text(d.failed));
        text(L"TokenTotals",L"Tokens   "+count_text(d.prefill_tokens)+L" prefill   ·   "+count_text(d.generated_tokens)+L" generated   ·   "+count_text(d.reused_tokens)+L" reused");
        auto seconds=static_cast<uint64_t>(d.uptime_seconds);
        text(L"SessionUptime",L"Uptime "+std::to_wstring(seconds/3600)+L":"+ (seconds/60%60<10?L"0":L"")+std::to_wstring(seconds/60%60)+L":"+(seconds%60<10?L"0":L"")+std::to_wstring(seconds%60));
        text(L"LatencyMetric",status.has_first_response?number(status.first_response_seconds,2)+L" s":L"—");
        text(L"LatencyScope",status.has_first_response?L"TTFT: "+(status.first_response_truncated?L"latest 4096":std::to_wstring(status.first_response_samples))+(status.first_response_samples==1?L" request · ":L" requests · ")+std::to_wstring(status.first_response_window_seconds)+L" s window":L"TTFT: no requests measured");
        ToolTipService::SetToolTip(control<TextBlock>(L"LatencyMetric"),box_value(L"Server-side time from accepting a request to offering its first nonempty answer or reasoning. Non-streaming requests are measured at body publication. The median uses retained samples in the stated time window."));
        text(L"KVCachePages",L"Pages   "+count_text(d.cached_pages)+L" cached · "+count_text(d.free_pages)+L" free · "+count_text(d.total_pages)+L" total");
        auto tokens=[&](uint64_t pages){return d.page_tokens&&pages>UINT64_MAX/d.page_tokens?UINT64_MAX:pages*d.page_tokens;};
        text(L"KVCacheTokens",L"Tokens   "+count_text(tokens(d.cached_pages))+L" cached · "+count_text(tokens(d.free_pages))+L" free · "+count_text(tokens(d.total_pages))+L" total");
        text(L"KVPageSize",count_text(d.page_tokens)+L" tokens/page");
        text(L"HostCacheHits",count_text(d.host_hits)+L" hits · "+count_text(d.host_reused_tokens)+L" reused tokens");
        text(L"HostCacheRecords",L"Records   "+count_text(d.host_memory_records)+L" in memory · "+count_text(d.host_disk_records)+L" on disk");
        text(L"HostCacheTransfers",L"Transfers   "+bytes(d.host_bytes_out)+L" out · "+bytes(d.host_bytes_in)+L" in");
        draw_chart();
    }
    void update_hardware(flare_snapshot const* snapshot,flare_status const& status) {
        auto count=flare_snapshot_gpu_count(snapshot);
        flare_gpu gpu{};
        if(count&&flare_snapshot_gpu(snapshot,0,&gpu)) {
            for(size_t i=0;i<count;++i){flare_gpu candidate{};if(flare_snapshot_gpu(snapshot,i,&candidate)&&candidate.has_visible_ordinal&&candidate.visible_ordinal==0){gpu=candidate;break;}}
            auto name=wide(gpu.name);auto annotation=name.find(L" (");if(annotation!=std::wstring::npos)name.resize(annotation);
            text(L"DeviceName",name);text(L"DeviceState",gpu.has_visible_ordinal?L"GPU detected":L"Not exposed to CUDA");
            if(gpu.has_total_memory&&gpu.has_free_memory&&gpu.total_memory_bytes) {
                auto used=gpu.total_memory_bytes-std::min(gpu.free_memory_bytes,gpu.total_memory_bytes);
                text(L"MemoryMetric",number(static_cast<double>(used)/(1ull<<30))+L" / "+number(static_cast<double>(gpu.total_memory_bytes)/(1ull<<30),0)+L" GiB");
                auto progress=control<ProgressBar>(L"MemoryProgress");progress.Visibility(Visibility::Visible);progress.Value(100.0*used/gpu.total_memory_bytes);
                text(L"MemoryNote",L"Driver-reported · all processes"+(status.has_hardware_sample_age?L" · "+number(status.hardware_sample_age_ms/1000.0,0)+L" s ago":L""));
            } else {text(L"MemoryMetric",gpu.has_total_memory?bytes(gpu.total_memory_bytes)+L" total":L"Not reported");control<ProgressBar>(L"MemoryProgress").Visibility(Visibility::Collapsed);text(L"MemoryNote",L"Free device memory is unavailable.");}
        } else {
            text(L"DeviceName",status.gpu_error.length?L"NVIDIA GPU unavailable":status.has_hardware_sample_age?L"No CUDA GPU detected":L"Checking this PC…");text(L"DeviceState",L"No device measurement");text(L"MemoryMetric",L"Not reported");control<ProgressBar>(L"MemoryProgress").Visibility(Visibility::Collapsed);text(L"MemoryNote",status.gpu_error.length?wide(status.gpu_error):L"Waiting for driver-reported device memory.");
        }
    }
